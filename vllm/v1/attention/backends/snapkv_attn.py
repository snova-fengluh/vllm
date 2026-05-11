# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SnapKV (Attention-Guided KV Cache Compression) Attention Backend for vLLM.

Algorithm overview
==================

SnapKV compresses the per-request KV cache *at the prefill→decode boundary*
using the attention pattern of the last ``query_window_size`` (W)
"observation" queries:

  1. For a request being prefilled (q_len == seq_len), once
     ``seq_len >= window_length``, slice the request's Q/K/V from the batch.
  2. Compute scores = softmax( Q[-W:] @ Kᵀ / sqrt(d) ) with a causal mask
     applied to the W×W block.
  3. Sum scores over the W observers for past positions [0 : seq_len - W].
  4. Apply 1-D pooling (avg or max, ``kernel_size`` odd) for spatial
     smoothing.
  5. Per query head, top-k select ``window_length - W`` past positions and
     gather the corresponding K/V slices.
  6. Concatenate ``[topk_K, last_W_K]`` (resp. V) and cache them in the
     per-request state ``compressed_keys_per_layer[layer_idx]``.

During *decode*, for any request whose prefill was compressed, the dense
KV is reconstructed as::

    dense_K = [ compressed_K (size = window_length) ]
              + [ paged_cache K from prefill_length .. current_seq_len ]

and standard dense FlashAttention (causal=False) runs against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.snapkv_cache_state import (
    SnapKVRequestState,
    SnapKVRequestStateRegistry,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import AttentionSpec

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )

logger = init_logger(__name__)

# SnapKV configuration defaults (can be overridden via SnapKVConfig)
SNAPKV_DEFAULT_WINDOW_LENGTH = 8192
SNAPKV_DEFAULT_QUERY_WINDOW_SIZE = 30
SNAPKV_DEFAULT_KERNEL_SIZE = 13
SNAPKV_DEFAULT_POOLING = "avgpool"
SNAPKV_DEFAULT_NUM_FULL_KV_LAYER = 0


# ---------------------------------------------------------------------------
# Backend descriptor
# ---------------------------------------------------------------------------


class SnapKVAttentionBackend(AttentionBackend):
    """SnapKV attention backend wrapping FlashAttention with bounded KV cache."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "SNAPKV_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    @staticmethod
    def get_impl_cls() -> type[SnapKVAttentionImpl]:
        return SnapKVAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[SnapKVAttentionMetadataBuilder]:
        return SnapKVAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        if is_fa_version_supported(4):
            return head_size <= 512
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        # SnapKV does not use sink tokens (StreamLLM-style); compression is
        # purely attention-guided.
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class SnapKVAttentionMetadata:
    """Per-batch metadata consumed by SnapKVAttentionImpl.forward()."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True

    # Per-row lists (length = num_reqs).
    request_ids: list[str] | None = None
    request_states: list[SnapKVRequestState | None] | None = None
    # Bool tensor [num_reqs] (CPU): True if request's state is compressed
    # (i.e. it has finished prefill compression and is in decode phase).
    is_compressed: torch.Tensor | None = None
    # Bool tensor [num_reqs] (CPU): True if this row is a prefill step
    # where observation queries should be saved for later compression.
    needs_save_obs_queries: torch.Tensor | None = None
    # Bool tensor [num_reqs] (CPU): True if this row is the first decode
    # step after prefill completed, triggering KV cache compression.
    needs_first_decode_compress: torch.Tensor | None = None
    # CPU int list/tensor of per-row seq_lens (Python ints) for fast access.
    seq_lens_cpu: torch.Tensor | None = None
    # CPU int list/tensor of per-row q_lens.
    q_lens_cpu: torch.Tensor | None = None
    # CPU int tensor [num_reqs+1] for slicing the batch per request without
    # forcing a GPU→CPU sync inside forward().
    query_start_loc_cpu: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


class SnapKVAttentionMetadataBuilder(
    AttentionMetadataBuilder[SnapKVAttentionMetadata]
):
    """Builder for SnapKV attention metadata."""

    # Disable CUDA graphs: SnapKV requires per-request top-k computation at
    # the end of prefill and dense KV reconstruction during decode for
    # variable-length compressed caches.
    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # Number of attention layers handled by *this* SnapKV builder.
        # When a request has populated its compressed_{keys,values}_per_layer
        # dict for all of these layers, prefill compression is finished and
        # the request can be flipped to ``compressed=True``.
        self.num_snap_layers = len(layer_names)

        self.num_heads_q = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        # SnapKV configuration (prefer snapkv_config from CLI, fall back to
        # attributes injected on hf_config).
        snap_cfg = vllm_config.snapkv_config
        if snap_cfg is not None:
            self.window_length = snap_cfg.window_length
            self.query_window_size = snap_cfg.query_window_size
            self.kernel_size = snap_cfg.kernel_size
            self.pooling = snap_cfg.pooling
            self.num_full_kv_layer = snap_cfg.num_full_kv_layer
        else:
            hf_config = self.model_config.hf_config
            self.window_length = getattr(
                hf_config, "snapkv_window_length",
                SNAPKV_DEFAULT_WINDOW_LENGTH,
            )
            self.query_window_size = getattr(
                hf_config, "snapkv_query_window_size",
                SNAPKV_DEFAULT_QUERY_WINDOW_SIZE,
            )
            self.kernel_size = getattr(
                hf_config, "snapkv_kernel_size",
                SNAPKV_DEFAULT_KERNEL_SIZE,
            )
            self.pooling = getattr(
                hf_config, "snapkv_pooling", SNAPKV_DEFAULT_POOLING,
            )
            self.num_full_kv_layer = getattr(
                hf_config, "snapkv_num_full_kv_layer",
                SNAPKV_DEFAULT_NUM_FULL_KV_LAYER,
            )

        self.topk_length = self.window_length - self.query_window_size

        # Per-request state registry
        self.state_registry = SnapKVRequestStateRegistry()

        logger.info(
            "SnapKV Attention initialized: window=%d, query_window=%d, "
            "kernel=%d, pooling=%s, topk=%d, full_kv_layers=%d",
            self.window_length,
            self.query_window_size,
            self.kernel_size,
            self.pooling,
            self.topk_length,
            self.num_full_kv_layer,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SnapKVAttentionMetadata:
        """Build SnapKV attention metadata."""
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal
        request_ids = common_attn_metadata.request_ids
        num_reqs = common_attn_metadata.num_reqs

        # Build per-row state lists.
        req_states: list[SnapKVRequestState | None] = []
        is_compressed_list: list[bool] = []
        needs_save_obs_list: list[bool] = []
        needs_first_decode_list: list[bool] = []
        seq_lens_list: list[int] = []
        q_lens_list: list[int] = []

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu_tensor = (
            seq_lens if seq_lens.is_cpu else seq_lens.cpu()
        )
        active_ids: set[str] = set()

        for i in range(num_reqs):
            rid = request_ids[i] if request_ids else ""
            if rid:
                active_ids.add(rid)
            seq_len_i = int(seq_lens_cpu_tensor[i].item())
            q_len_i = int(query_start_loc_cpu[i + 1] - query_start_loc_cpu[i])
            seq_lens_list.append(seq_len_i)
            q_lens_list.append(q_len_i)

            state = self.state_registry.get_or_create(rid) if rid else None

            # If the state finished collecting per-layer compressed K/V in
            # the *previous* step's forward(), flip it to compressed now so
            # that this step's forward() takes the dense-reconstruction path.
            if (
                state is not None
                and not state.compressed
                and self.num_snap_layers > 0
                and len(state.compressed_keys_per_layer) >= self.num_snap_layers
            ):
                state.compressed = True
                logger.info(
                    "SnapKV STATE FLIP request=%r: compressed=True "
                    "(%d/%d layers populated, prefill_length=%d)",
                    rid, len(state.compressed_keys_per_layer),
                    self.num_snap_layers, state.prefill_length,
                )

            is_prefill_step = q_len_i > 1
            is_decode_step = q_len_i == 1

            # Track prefill progress for chunked-prefill support.
            if is_prefill_step and state is not None:
                state.in_prefill = True
                state.prefill_seq_len = seq_len_i

            # During prefill: save observation queries (last W) per layer so
            # compression can run at the first decode step.
            needs_save_obs = (
                is_prefill_step
                and state is not None
                and not state.compressed
            )
            needs_save_obs_list.append(needs_save_obs)

            # First decode step after prefill: trigger compression using
            # saved observation queries + K/V gathered from paged cache.
            needs_first_decode = (
                is_decode_step
                and state is not None
                and state.in_prefill
                and not state.compressed
                and state.prefill_seq_len >= self.window_length
            )
            if needs_first_decode and state is not None:
                state.in_prefill = False  # consumed
            needs_first_decode_list.append(needs_first_decode)

            is_compressed = state is not None and state.compressed
            is_compressed_list.append(is_compressed)

            req_states.append(state)

        # Prune finished requests.
        if active_ids:
            self.state_registry.prune(active_ids)

        # CPU-resident control tensors (avoid GPU↔CPU sync inside forward).
        is_compressed = torch.tensor(
            is_compressed_list, dtype=torch.bool, device="cpu"
        )
        needs_save_obs_queries = torch.tensor(
            needs_save_obs_list, dtype=torch.bool, device="cpu"
        )
        needs_first_decode_compress = torch.tensor(
            needs_first_decode_list, dtype=torch.bool, device="cpu"
        )
        seq_lens_cpu = torch.tensor(
            seq_lens_list, dtype=torch.int64, device="cpu"
        )
        q_lens_cpu = torch.tensor(
            q_lens_list, dtype=torch.int64, device="cpu"
        )

        return SnapKVAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=causal,
            request_ids=request_ids,
            request_states=req_states,
            is_compressed=is_compressed,
            needs_save_obs_queries=needs_save_obs_queries,
            needs_first_decode_compress=needs_first_decode_compress,
            seq_lens_cpu=seq_lens_cpu,
            q_lens_cpu=q_lens_cpu,
            query_start_loc_cpu=query_start_loc_cpu.clone(),
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, n_rep: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand KV from kv_heads to query_heads by repeating (GQA expansion).

    Args:
        keys:   [seq_len, num_kv_heads, head_dim]
        values: [seq_len, num_kv_heads, head_dim]
        n_rep:  num_query_heads // num_kv_heads

    Returns:
        (keys, values) each [seq_len, num_query_heads, head_dim]
    """
    if n_rep == 1:
        return keys, values
    seq_len, num_kv_heads, head_dim = keys.shape
    keys = keys.unsqueeze(2).expand(seq_len, num_kv_heads, n_rep, head_dim)
    values = values.unsqueeze(2).expand(seq_len, num_kv_heads, n_rep, head_dim)
    keys = keys.reshape(seq_len, num_kv_heads * n_rep, head_dim).contiguous()
    values = values.reshape(seq_len, num_kv_heads * n_rep, head_dim).contiguous()
    return keys, values


def _resolve_layer_index(layer: torch.nn.Module) -> int:
    """Resolve the integer layer index for an attention layer.

    First checks for an explicit ``_layer_index`` attribute (set by some
    model wrappers); falls back to parsing the layer's ``layer_name``
    (e.g. ``model.layers.7.self_attn.attn``) via ``extract_layer_index``.
    """
    explicit = getattr(layer, "_layer_index", None)
    if isinstance(explicit, int):
        return explicit
    layer_name = getattr(layer, "layer_name", None)
    if layer_name is None:
        return 0
    from vllm.model_executor.models.utils import extract_layer_index

    try:
        return int(extract_layer_index(layer_name))
    except Exception:
        return 0


def _gather_paged_kv(
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather KV pairs from the paged cache for a single request.

    Args:
        kv_cache:        [2, num_blocks, block_size, num_kv_heads, head_size]
        block_table_row: [max_blocks_per_seq] int32 block ids for this req.
        token_indices:   1-D int64 tensor of token positions to gather.
        block_size:      tokens per block.

    Returns:
        (keys, values) each [len(token_indices), num_kv_heads, head_size]
    """
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]
    block_ids = block_table_row[token_indices // block_size]
    slot_offsets = token_indices % block_size
    return key_cache[block_ids, slot_offsets], value_cache[block_ids, slot_offsets]


def _gather_paged_keys(
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather only keys (not values) from the paged cache.

    Same as _gather_paged_kv but avoids materializing the value tensor,
    halving peak memory during the scoring phase of compression.

    Returns:
        keys [len(token_indices), num_kv_heads, head_size]
    """
    key_cache = kv_cache[0]
    block_ids = block_table_row[token_indices // block_size]
    slot_offsets = token_indices % block_size
    return key_cache[block_ids, slot_offsets]


def _snapkv_compute_topk(
    q: torch.Tensor,        # [seq_len, num_q_heads, head_dim]
    k: torch.Tensor,        # [seq_len, num_q_heads, head_dim]  (already GQA-expanded)
    v: torch.Tensor,        # [seq_len, num_q_heads, head_dim]
    query_window_size: int,
    topk_length: int,
    kernel_size: int,
    pooling: str,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SnapKV's prefill compression for one request.

    Returns:
        (compressed_keys, compressed_values) each
            [num_q_heads, topk_length + query_window_size, head_dim]
        i.e. [topk_past_K, last_W_K] concatenated along sequence dim.
    """
    seq_len, num_q_heads, head_dim = q.shape
    W = query_window_size

    # Permute to head-major for batched matmul
    # q: [H, S, D], k: [H, S, D], v: [H, S, D]
    qh = q.permute(1, 0, 2).contiguous()
    kh = k.permute(1, 0, 2).contiguous()
    vh = v.permute(1, 0, 2).contiguous()

    # If there is no room for past tokens, short-circuit and return just
    # the observation window. This also avoids pooling over a zero-length
    # axis, which is invalid for F.avg_pool1d / F.max_pool1d.
    if seq_len - W <= 0 or topk_length <= 0:
        last_w_k = kh[:, -W:, :]                          # [H, W, D]
        last_w_v = vh[:, -W:, :]
        return last_w_k.contiguous(), last_w_v.contiguous()

    # Use only the last W queries as observers.
    q_obs = qh[:, -W:, :]                                # [H, W, D]

    # scores: [H, W, S]
    scores = torch.matmul(q_obs, kh.transpose(1, 2)) * scale

    # Apply causal mask to the trailing W×W block. For observer i (relative
    # to start-of-window) and key j (relative to start-of-window),
    # mask off positions where j > i.
    neg_inf = torch.finfo(scores.dtype).min
    causal_mask = torch.full(
        (W, W), neg_inf, device=scores.device, dtype=scores.dtype
    )
    idx = torch.arange(W, device=scores.device)
    causal_mask.masked_fill_(idx.unsqueeze(0) <= idx.unsqueeze(1), 0)
    scores[:, :, -W:] = scores[:, :, -W:] + causal_mask  # broadcast over H

    # Softmax over keys. Use float32 internally for numerical stability.
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    # Sum over the W observers, restricted to past positions [0, S-W).
    # weights[:, :, :S-W] -> sum over dim=1 -> [H, S-W]
    weights_sum = weights[:, :, : seq_len - W].sum(dim=1)

    # 1-D pooling over the past-position axis.
    pad = kernel_size // 2
    if pooling == "avgpool":
        pooled = F.avg_pool1d(
            weights_sum, kernel_size=kernel_size, padding=pad, stride=1
        )
    elif pooling == "maxpool":
        pooled = F.max_pool1d(
            weights_sum, kernel_size=kernel_size, padding=pad, stride=1
        )
    else:
        raise ValueError(f"Unknown SnapKV pooling: {pooling!r}")
    # pooled: [H, S-W]

    # Top-k indices per query head over past positions.
    actual_k = min(topk_length, seq_len - W)
    if actual_k <= 0:
        # No room for any past tokens; return only the observation window.
        last_w_k = kh[:, -W:, :]                          # [H, W, D]
        last_w_v = vh[:, -W:, :]
        return last_w_k.contiguous(), last_w_v.contiguous()

    _, indices = torch.topk(pooled, k=actual_k, dim=-1, largest=True, sorted=False)
    # indices: [H, actual_k]
    indices_exp = indices.unsqueeze(-1).expand(-1, -1, head_dim)

    k_past = kh[:, : seq_len - W, :]                      # [H, S-W, D]
    v_past = vh[:, : seq_len - W, :]
    topk_k = torch.gather(k_past, dim=1, index=indices_exp)  # [H, k, D]
    topk_v = torch.gather(v_past, dim=1, index=indices_exp)

    # Concat with observation window.
    last_w_k = kh[:, -W:, :]                               # [H, W, D]
    last_w_v = vh[:, -W:, :]

    compressed_k = torch.cat([topk_k, last_w_k], dim=1)    # [H, k+W, D]
    compressed_v = torch.cat([topk_v, last_w_v], dim=1)
    return compressed_k.contiguous(), compressed_v.contiguous()


def _snapkv_compute_topk_indices(
    obs_q: torch.Tensor,    # [W, num_q_heads, head_dim] (observation queries)
    k: torch.Tensor,        # [seq_len, num_q_heads, head_dim] (GQA-expanded)
    seq_len: int,
    query_window_size: int,
    topk_length: int,
    kernel_size: int,
    pooling: str,
    scale: float,
) -> torch.Tensor | None:
    """Compute only the top-k past-position indices (no V needed).

    This is the scoring-only half of _snapkv_compute_topk, designed for
    memory-efficient first-decode compression where we want to avoid
    materializing V during index selection.

    Args:
        obs_q:  Observation queries [W, num_q_heads, head_dim].
        k:      Full keys [seq_len, num_q_heads, head_dim] (GQA-expanded).
        seq_len: Total prefill sequence length.
        query_window_size: W — number of observation queries.
        topk_length: Number of past positions to select.
        kernel_size: 1-D pooling kernel size (odd).
        pooling: "avgpool" or "maxpool".
        scale:  Attention scale factor.

    Returns:
        indices [num_q_heads, actual_k] of selected past positions, or
        None if there are no past tokens to select from.
    """
    W = query_window_size

    if seq_len - W <= 0 or topk_length <= 0:
        return None

    # Permute to head-major: [H, *, D]
    q_obs = obs_q.permute(1, 0, 2).contiguous()          # [H, W, D]
    kh = k.permute(1, 0, 2).contiguous()                 # [H, S, D]

    # Attention scores: [H, W, S]
    scores = torch.matmul(q_obs, kh.transpose(1, 2)) * scale

    # Causal mask on the W×W observation block.
    neg_inf = torch.finfo(scores.dtype).min
    causal_mask = torch.full(
        (W, W), neg_inf, device=scores.device, dtype=scores.dtype,
    )
    idx = torch.arange(W, device=scores.device)
    causal_mask.masked_fill_(idx.unsqueeze(0) <= idx.unsqueeze(1), 0)
    scores[:, :, -W:] = scores[:, :, -W:] + causal_mask

    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(obs_q.dtype)
    del scores  # free [H, W, S] immediately

    # Sum over observers for past positions [0, S-W).
    weights_sum = weights[:, :, : seq_len - W].sum(dim=1)  # [H, S-W]
    del weights

    # 1-D pooling.
    pad = kernel_size // 2
    if pooling == "avgpool":
        pooled = F.avg_pool1d(
            weights_sum, kernel_size=kernel_size, padding=pad, stride=1,
        )
    elif pooling == "maxpool":
        pooled = F.max_pool1d(
            weights_sum, kernel_size=kernel_size, padding=pad, stride=1,
        )
    else:
        raise ValueError(f"Unknown SnapKV pooling: {pooling!r}")
    del weights_sum

    actual_k = min(topk_length, seq_len - W)
    if actual_k <= 0:
        return None

    _, indices = torch.topk(pooled, k=actual_k, dim=-1, largest=True, sorted=False)
    # indices: [H, actual_k]
    return indices


# ---------------------------------------------------------------------------
# Attention implementation
# ---------------------------------------------------------------------------


class SnapKVAttentionImpl(AttentionImpl):
    """SnapKV attention.

    - Prefill: standard FlashAttention; at the end of forward, for any
      request whose prefill ended with seq_len >= window_length, run the
      SnapKV top-k compression and store the compressed K/V in the
      per-request state.
    - Decode (request not yet compressed): standard paged FlashAttention.
    - Decode (request compressed): build dense KV from
      [stored compressed K/V] + [paged cache rows from prefill_length to
      current_seq_len] and run dense FA (causal=False).
    """

    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        if alibi_slopes is not None:
            raise ValueError("SnapKV attention does not support ALiBi")

        self.sliding_window = (-1, -1)
        self.kv_cache_dtype = kv_cache_dtype

        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.attn_type = attn_type

        self.vllm_flash_attn_version = get_flash_attn_version(head_size=head_size)

        logger.info_once(
            "SnapKV attention using FlashAttention version %s",
            self.vllm_flash_attn_version,
        )

        # SnapKV config
        vllm_config = get_current_vllm_config()
        snap_cfg = vllm_config.snapkv_config
        if snap_cfg is not None:
            self.window_length = snap_cfg.window_length
            self.query_window_size = snap_cfg.query_window_size
            self.kernel_size = snap_cfg.kernel_size
            self.pooling = snap_cfg.pooling
            self.num_full_kv_layer = snap_cfg.num_full_kv_layer
        else:
            hf_config = vllm_config.model_config.hf_config
            self.window_length = getattr(
                hf_config, "snapkv_window_length",
                SNAPKV_DEFAULT_WINDOW_LENGTH,
            )
            self.query_window_size = getattr(
                hf_config, "snapkv_query_window_size",
                SNAPKV_DEFAULT_QUERY_WINDOW_SIZE,
            )
            self.kernel_size = getattr(
                hf_config, "snapkv_kernel_size",
                SNAPKV_DEFAULT_KERNEL_SIZE,
            )
            self.pooling = getattr(
                hf_config, "snapkv_pooling", SNAPKV_DEFAULT_POOLING,
            )
            self.num_full_kv_layer = getattr(
                hf_config, "snapkv_num_full_kv_layer",
                SNAPKV_DEFAULT_NUM_FULL_KV_LAYER,
            )

        self.topk_length = self.window_length - self.query_window_size

    # ---- KV cache update (called by runner before forward) ----

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    # ---- Forward ----

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SnapKVAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        assert self.vllm_flash_attn_version is not None

        layer_idx = _resolve_layer_index(layer)

        # ------------------------------------------------------------------
        # Phase 1: During prefill chunks, save the last W observation queries
        # per request per layer for later compression.
        # ------------------------------------------------------------------
        if (
            attn_metadata.needs_save_obs_queries is not None
            and bool(attn_metadata.needs_save_obs_queries.any().item())
        ):
            self._save_observation_queries(query, attn_metadata, layer_idx)

        # ------------------------------------------------------------------
        # Phase 2: At the first decode step after prefill completes, gather
        # full K/V from the paged cache and run SnapKV top-k compression.
        # ------------------------------------------------------------------
        if (
            attn_metadata.needs_first_decode_compress is not None
            and bool(attn_metadata.needs_first_decode_compress.any().item())
        ):
            n_compress = int(
                attn_metadata.needs_first_decode_compress.sum().item()
            )
            logger.info(
                "SnapKV FIRST-DECODE COMPRESS layer=%d: %d request(s) "
                "triggering top-k compression from paged cache",
                layer_idx, n_compress,
            )
            self._do_first_decode_compression(
                kv_cache, attn_metadata, layer_idx
            )
            # Mark newly-compressed requests so Phase 3 routes them through
            # the compressed decode path on this same step (consistent with
            # the sambaLLMs reference where the first decode token already
            # uses compressed KV).
            attn_metadata.is_compressed = (
                attn_metadata.is_compressed
                | attn_metadata.needs_first_decode_compress
            )

        # ------------------------------------------------------------------
        # Phase 3: Run attention.
        # ------------------------------------------------------------------
        has_compressed_decode = (
            attn_metadata.is_compressed is not None
            and attn_metadata.max_query_len == 1
            and bool(attn_metadata.is_compressed.any().item())
        )

        if has_compressed_decode:
            n_comp = int(attn_metadata.is_compressed.sum().item())
            n_total = int(attn_metadata.is_compressed.shape[0])
            logger.info(
                "SnapKV DECODE layer=%d: %d/%d requests using compressed KV",
                layer_idx, n_comp, n_total,
            )
            self._decode_mixed_forward(
                layer, query, kv_cache, attn_metadata, output, layer_idx
            )
        else:
            self._flash_attn_forward(
                layer, query, kv_cache, attn_metadata, output
            )

        return output

    # ---- Standard FA path (used for prefill and short-context decode) ----

    def _flash_attn_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SnapKVAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)

        descale_shape = (
            attn_metadata.query_start_loc.shape[0] - 1,
            self.num_kv_heads,
        )
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=None,
            window_size=list(self.sliding_window),
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return output

    # ---- Decode: mix of compressed and uncompressed rows ----

    def _decode_mixed_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SnapKVAttentionMetadata,
        output: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Handle a decode batch mixing compressed and standard rows.

        Compressed rows (``is_compressed == True``): build a dense KV from
        ``[stored compressed]`` + ``[paged cache for tokens after prefill]``
        and run dense FA (causal=False).

        Uncompressed rows: standard paged FA.
        """
        device = query.device

        comp_mask = attn_metadata.is_compressed         # CPU bool [num_reqs]
        std_mask = ~comp_mask
        comp_row_cpu = comp_mask.nonzero(as_tuple=True)[0]
        std_row_cpu = std_mask.nonzero(as_tuple=True)[0]

        N_comp = int(comp_row_cpu.shape[0])
        N_std = int(std_row_cpu.shape[0])

        # ---- Standard rows: one batched paged-FA call ----
        if N_std > 0:
            self._batched_standard_decode(
                layer, query, kv_cache, attn_metadata, output,
                std_row_cpu, N_std,
            )

        # ---- Compressed rows: per-row dense FA against compressed + tail ----
        if N_comp > 0:
            self._batched_compressed_decode(
                layer, query, kv_cache, attn_metadata, output,
                comp_row_cpu, N_comp, layer_idx, device,
            )

    def _batched_standard_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SnapKVAttentionMetadata,
        output: torch.Tensor,
        std_row_cpu: torch.Tensor,
        N_std: int,
    ) -> None:
        """One FA kernel call for all uncompressed-decode rows."""
        device = query.device
        key_cache, value_cache = kv_cache.unbind(0)

        std_gpu = std_row_cpu.to(device, dtype=torch.long, non_blocking=True)

        q_std = query[std_gpu]                                  # [N, H_q, D]
        seq_lens_std = attn_metadata.seq_lens[std_gpu]
        block_table_std = attn_metadata.block_table[std_gpu]

        cu_seqlens_q = torch.arange(
            0, N_std + 1, dtype=torch.int32, device=device
        )
        max_seqlen_k = int(seq_lens_std.max().item())

        descale_shape = (N_std, self.num_kv_heads)
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        out_std = torch.empty_like(q_std)
        flash_attn_varlen_func(
            q=q_std,
            k=key_cache,
            v=value_cache,
            out=out_std,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seq_lens_std,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=None,
            window_size=list(self.sliding_window),
            block_table=block_table_std,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        output[std_gpu] = out_std

    def _batched_compressed_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SnapKVAttentionMetadata,
        output: torch.Tensor,
        comp_row_cpu: torch.Tensor,
        N: int,
        layer_idx: int,
        device: torch.device,
    ) -> None:
        """Build dense KV and run dense FA for compressed-cache decode rows.

        For each compressed row i:
            seq_len_i  = current sequence length
            prefill_i  = state.prefill_length
            compressed = [num_q_heads, window_length, head_dim]
                         (prefill snapshot of size window_length)
            tail = paged_cache[block_table[i], prefill_i : seq_len_i]
                         (decode tokens added since prefill)
            dense_KV   = [compressed | tail]   # along seq dim
        Run dense FA with causal=False over dense_KV.

        Per-row dense lengths can differ (different decode progress), so
        we pad to the per-batch maximum for the varlen call.
        """
        head_dim = self.head_size
        H_q = self.num_heads
        block_size = kv_cache.shape[2]
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # Per-row Python data
        per_row_dense_k: list[torch.Tensor] = []
        per_row_dense_v: list[torch.Tensor] = []
        per_row_lens: list[int] = []

        for idx in range(N):
            row_i = int(comp_row_cpu[idx])
            state = attn_metadata.request_states[row_i]
            assert state is not None
            assert layer_idx in state.compressed_keys_per_layer, (
                f"SnapKV decode: layer {layer_idx} missing compressed K "
                f"for request idx {row_i}"
            )

            # Transfer compressed K/V from CPU to GPU just-in-time.
            comp_k = state.compressed_keys_per_layer[layer_idx].to(
                device, non_blocking=True,
            )   # [H_q, W, D]
            comp_v = state.compressed_values_per_layer[layer_idx].to(
                device, non_blocking=True,
            )
            window_len = comp_k.shape[1]

            seq_len_i = int(attn_metadata.seq_lens_cpu[row_i].item())
            prefill_len = state.prefill_length
            tail_len = max(0, seq_len_i - prefill_len)

            block_table_row = attn_metadata.block_table[row_i]

            if tail_len > 0:
                tail_idx = torch.arange(
                    prefill_len, seq_len_i, dtype=torch.int64, device=device
                )
                tail_k_kv, tail_v_kv = _gather_paged_kv(
                    kv_cache, block_table_row, tail_idx, block_size,
                )
                # GQA expand to query head level.
                tail_k, tail_v = _repeat_kv(
                    tail_k_kv, tail_v_kv, self.num_queries_per_kv
                )
                # Permute to [H_q, tail_len, D] to match compressed layout.
                tail_k = tail_k.permute(1, 0, 2).contiguous()
                tail_v = tail_v.permute(1, 0, 2).contiguous()
                dense_k = torch.cat([comp_k, tail_k], dim=1)
                dense_v = torch.cat([comp_v, tail_v], dim=1)
            else:
                dense_k = comp_k
                dense_v = comp_v

            per_row_dense_k.append(dense_k)
            per_row_dense_v.append(dense_v)
            per_row_lens.append(int(dense_k.shape[1]))

        # Build varlen inputs by concatenating along the seq dim.
        # Each row contributes per_row_lens[i] entries; we go to layout
        # [sum_lens, H_q, D].
        comp_gpu = comp_row_cpu.to(device, dtype=torch.long, non_blocking=True)

        # Permute each [H_q, L_i, D] -> [L_i, H_q, D] then cat.
        cat_k = torch.cat(
            [t.permute(1, 0, 2).contiguous() for t in per_row_dense_k], dim=0
        )
        cat_v = torch.cat(
            [t.permute(1, 0, 2).contiguous() for t in per_row_dense_v], dim=0
        )
        # cat_k: [sum_L, H_q, D]

        cu_seqlens_k_list = [0]
        running = 0
        for L in per_row_lens:
            running += L
            cu_seqlens_k_list.append(running)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k_list, dtype=torch.int32, device=device
        )
        max_seqlen_k = max(per_row_lens) if per_row_lens else 1

        cu_seqlens_q = torch.arange(
            0, N + 1, dtype=torch.int32, device=device
        )

        q_comp = query[comp_gpu]                              # [N, H_q, D]
        out_comp = torch.empty_like(q_comp)

        flash_attn_varlen_func(
            q=q_comp,
            k=cat_k,
            v=cat_v,
            out=out_comp,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=None,
            window_size=[-1, -1],
            block_table=None,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
        )
        output[comp_gpu] = out_comp

    # ---- Observation-query saving (during prefill chunks) ----

    def _save_observation_queries(
        self,
        query: torch.Tensor,
        attn_metadata: SnapKVAttentionMetadata,
        layer_idx: int,
    ) -> None:
        """Save the last W observation queries from each prefill chunk.

        During chunked prefill the full prefill Q is never available in a
        single batch.  Instead, each chunk saves its last W queries (or
        fewer if the chunk is shorter than W).  Across chunks the saved
        tensor is accumulated so that when prefill ends the most recent W
        queries of the entire prefill are available for compression.
        """
        if layer_idx < self.num_full_kv_layer:
            return

        needs = attn_metadata.needs_save_obs_queries
        rows = needs.nonzero(as_tuple=True)[0].tolist()
        if not rows:
            return

        W = self.query_window_size

        for row_i in rows:
            row_i = int(row_i)
            state = attn_metadata.request_states[row_i]
            if state is None or state.compressed:
                continue

            start = int(attn_metadata.query_start_loc_cpu[row_i].item())
            q_len_i = int(attn_metadata.q_lens_cpu[row_i].item())
            end = start + q_len_i

            q_chunk = query[start:end]        # [q_len_i, H_q, D]
            obs_new = q_chunk[-min(W, q_len_i):]  # last W (or fewer)

            # Accumulate across chunks to always keep the last W queries.
            prev = state.observation_queries_per_layer.get(layer_idx)
            if prev is not None and obs_new.shape[0] < W:
                combined = torch.cat([prev, obs_new], dim=0)
                obs_new = combined[-W:]

            state.observation_queries_per_layer[layer_idx] = obs_new.clone()

    # ---- First-decode compression (from paged cache) ----

    def _do_first_decode_compression(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: SnapKVAttentionMetadata,
        layer_idx: int,
    ) -> None:
        """At the first decode step, gather K/V from paged cache and compress.

        Memory-efficient implementation:
          1. Gather only K (not V) from paged cache for scoring.
          2. Compute top-k indices using _snapkv_compute_topk_indices.
          3. Gather only the selected top-k + observation window positions'
             K/V from paged cache (much smaller than the full gather).
          4. Store compressed K/V on CPU to avoid GPU memory accumulation
             across all layers (~9 GB for 62 layers x 3 requests on GPU).
        """
        if layer_idx < self.num_full_kv_layer:
            return

        needs = attn_metadata.needs_first_decode_compress
        rows = needs.nonzero(as_tuple=True)[0].tolist()
        if not rows:
            return

        device = kv_cache.device
        block_size = kv_cache.shape[2]
        num_kv_heads = kv_cache.shape[3]
        head_dim = kv_cache.shape[4]
        num_q_heads = self.num_heads

        for row_i in rows:
            row_i = int(row_i)
            state = attn_metadata.request_states[row_i]
            if state is None or state.compressed:
                continue

            obs_q = state.observation_queries_per_layer.get(layer_idx)
            if obs_q is None:
                logger.warning(
                    "SnapKV: no observation queries for layer %d, "
                    "request row %d — skipping compression",
                    layer_idx, row_i,
                )
                continue

            prefill_len = state.prefill_seq_len
            actual_w = obs_q.shape[0]
            topk_length = self.window_length - actual_w
            block_table_row = attn_metadata.block_table[row_i]

            # ----------------------------------------------------------
            # Step 1: Gather only K for scoring (half the memory of K+V).
            # ----------------------------------------------------------
            all_indices = torch.arange(
                prefill_len, dtype=torch.int64, device=device,
            )
            k_all = _gather_paged_keys(
                kv_cache, block_table_row, all_indices, block_size,
            )
            # k_all: [prefill_len, num_kv_heads, head_dim]
            del all_indices

            # GQA expand K only.
            if self.num_queries_per_kv > 1:
                k_expanded = k_all.unsqueeze(2).expand(
                    prefill_len, num_kv_heads, self.num_queries_per_kv,
                    head_dim,
                ).reshape(prefill_len, num_q_heads, head_dim).contiguous()
                del k_all
            else:
                k_expanded = k_all

            # ----------------------------------------------------------
            # Step 2: Compute top-k indices (no V materialized).
            # ----------------------------------------------------------
            # obs_q is [actual_w, num_q_heads, head_dim] on GPU.
            indices = _snapkv_compute_topk_indices(
                obs_q, k_expanded, prefill_len,
                query_window_size=actual_w,
                topk_length=topk_length,
                kernel_size=self.kernel_size,
                pooling=self.pooling,
                scale=self.scale,
            )
            del k_expanded  # free the large expanded K tensor

            # ----------------------------------------------------------
            # Step 3: Gather selected positions from paged cache.
            # Each query head independently selected its own top-k
            # past positions. We build per-head position lists,
            # flatten into a single paged cache gather, then extract
            # each query head's corresponding KV head.
            # ----------------------------------------------------------
            obs_window_start = prefill_len - actual_w
            obs_positions = torch.arange(
                obs_window_start, prefill_len,
                dtype=torch.int64, device=device,
            )
            # Which KV head each query head maps to under GQA.
            kv_head_idx = torch.arange(
                num_q_heads, device=device,
            ) // self.num_queries_per_kv  # [H_q]

            if indices is not None:
                # indices: [num_q_heads, actual_k] — positions in [0, S-W)
                actual_k = indices.shape[1]
                total_per_head = actual_k + actual_w

                # Per-head position list: [past_selected | obs_window].
                # all_positions: [H_q, actual_k + actual_w]
                all_positions = torch.cat([
                    indices.long(),
                    obs_positions.unsqueeze(0).expand(num_q_heads, -1),
                ], dim=1)

                # Single gather from paged cache.
                flat_positions = all_positions.reshape(-1)
                k_flat, v_flat = _gather_paged_kv(
                    kv_cache, block_table_row, flat_positions, block_size,
                )
                # k_flat, v_flat: [H_q * total_per_head, num_kv_heads, D]
                del flat_positions, all_positions

                # Reshape to per-head: [H_q, total_per_head, num_kv_heads, D]
                k_per_head = k_flat.reshape(
                    num_q_heads, total_per_head, num_kv_heads, head_dim,
                )
                v_per_head = v_flat.reshape(
                    num_q_heads, total_per_head, num_kv_heads, head_dim,
                )
                del k_flat, v_flat

                # Extract each query head's mapped KV head via gather.
                kv_sel = kv_head_idx.reshape(
                    num_q_heads, 1, 1, 1,
                ).expand(-1, total_per_head, 1, head_dim)
                comp_k = torch.gather(
                    k_per_head, dim=2, index=kv_sel,
                ).squeeze(2)  # [H_q, total_per_head, D]
                comp_v = torch.gather(
                    v_per_head, dim=2, index=kv_sel,
                ).squeeze(2)
                del k_per_head, v_per_head, kv_sel
            else:
                # No past tokens to select; use only observation window.
                actual_k = 0
                k_obs, v_obs = _gather_paged_kv(
                    kv_cache, block_table_row, obs_positions, block_size,
                )
                # k_obs, v_obs: [actual_w, num_kv_heads, head_dim]
                # Extract each query head's KV head: [actual_w, H_q, D]
                comp_k = k_obs[:, kv_head_idx, :].permute(
                    1, 0, 2,
                ).contiguous()  # [H_q, actual_w, D]
                comp_v = v_obs[:, kv_head_idx, :].permute(
                    1, 0, 2,
                ).contiguous()
                del k_obs, v_obs

            del obs_positions

            # ----------------------------------------------------------
            # Step 4: Store on CPU to avoid GPU memory accumulation.
            # ----------------------------------------------------------
            state.compressed_keys_per_layer[layer_idx] = comp_k.cpu()
            state.compressed_values_per_layer[layer_idx] = comp_v.cpu()
            state.prefill_length = prefill_len
            del comp_k, comp_v

            logger.info(
                "SnapKV COMPRESSED row=%d layer=%d: "
                "prefill_seq_len=%d -> compressed shape=[%d, %d, %d] "
                "(topk=%d + obs_window=%d), prefill_length=%d, "
                "stored on CPU",
                row_i, layer_idx, prefill_len,
                num_q_heads, actual_k + actual_w, head_dim,
                actual_k, actual_w, state.prefill_length,
            )

            # Free observation queries for this layer (no longer needed).
            state.observation_queries_per_layer.pop(layer_idx, None)

        # The state's ``compressed`` flag is flipped by the metadata
        # builder on the *next* step, once all SnapKV layers have stored
        # their per-layer compressed K/V (see
        # ``SnapKVAttentionMetadataBuilder.build``).


