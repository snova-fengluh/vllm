# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SAGE (Self-Attention Guided Eviction) Attention Backend for vLLM.

Algorithm (per decode row whose seq_len > window_length):
  1. Gather candidate KV slice [num_sink : seq_len - recent_window] from
     the paged cache.
  2. Compute scores = query · keysᵀ · scale, GQA-reduce, topk → indices.
     Store indices on SageRequestState (done once at prefill→decode edge).
  3. Build dense KV = concat(paged[0:num_sink],
                             paged[topk_indices],
                             paged[recent_start : seq_len])
  4. Run flash_attn_varlen_func against the dense KV (causal=False).

Recent-window policy (controlled by SageConfig.growing_recent_window):
  * False (default, StreamingLLM-style): the recent slice for each request
    is always the last ``recent_window`` tokens, i.e.
    ``recent_start = seq_len - recent_window``. The oldest recent token is
    evicted each decode step; recent length is constant.
  * True (growing): ``recent_start`` is frozen at the first long-decode step
    to ``seq_len - recent_window`` and reused thereafter. The recent slice
    is ``[recent_start, seq_len - 1]`` and grows by one token per decode
    step. Memory cost per long-context request grows linearly with the
    number of generated tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

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
from vllm.v1.attention.backends.sage_cache_state import (
    SageRequestState,
    SageRequestStateRegistry,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import AttentionSpec

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )

logger = init_logger(__name__)

# SAGE configuration defaults (can be overridden via model config)
SAGE_DEFAULT_WINDOW_LENGTH = 8192
SAGE_DEFAULT_NUM_SINK_TOKENS = 4
SAGE_DEFAULT_TOP_K = 512


# ---------------------------------------------------------------------------
# Backend descriptor
# ---------------------------------------------------------------------------


class SageAttentionBackend(AttentionBackend):
    """SAGE attention backend wrapping FlashAttention with bounded KV cache."""

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
        return "SAGE_ATTN"

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
    def get_impl_cls() -> type[SageAttentionImpl]:
        return SageAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[SageAttentionMetadataBuilder]:
        return SageAttentionMetadataBuilder

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
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class SageAttentionMetadata:
    """Per-batch metadata consumed by SageAttentionImpl.forward()."""

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
    request_states: list[SageRequestState | None] | None = None
    # Bool tensor [num_reqs]: True when seq_len > window_length.
    is_long_context: torch.Tensor | None = None
    # Bool tensor [num_reqs]: True on the first decode step after prefill
    # for that request.
    is_first_long_decode: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


class SageAttentionMetadataBuilder(AttentionMetadataBuilder[SageAttentionMetadata]):
    """Builder for SAGE attention metadata."""

    # SAGE requires dynamic execution for long-context decodes (variable dense
    # KV sizes, top-k selection, etc.), so we disable CUDA graphs entirely.
    # This has some performance impact for short-context batches, but is
    # necessary for SAGE to function correctly.
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

        self.num_heads_q = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        # SAGE configuration (prefer sage_config from CLI, fall back to
        # attributes injected on hf_config).
        sage_cfg = vllm_config.sage_config
        if sage_cfg is not None:
            self.sage_window_length = sage_cfg.window_length
            self.sage_num_sink_tokens = sage_cfg.num_sink_tokens
            self.sage_top_k = sage_cfg.top_k
            self.sage_growing_recent_window = sage_cfg.growing_recent_window
        else:
            hf_config = self.model_config.hf_config
            self.sage_window_length = getattr(
                hf_config, "sage_window_length", SAGE_DEFAULT_WINDOW_LENGTH
            )
            self.sage_num_sink_tokens = getattr(
                hf_config, "sage_num_sink_tokens", SAGE_DEFAULT_NUM_SINK_TOKENS
            )
            self.sage_top_k = getattr(hf_config, "sage_top_k", SAGE_DEFAULT_TOP_K)
            self.sage_growing_recent_window = getattr(
                hf_config, "sage_growing_recent_window", False
            )

        self.sage_recent_window = (
            self.sage_window_length - self.sage_num_sink_tokens - self.sage_top_k
        )

        # Per-request state registry
        self.state_registry = SageRequestStateRegistry()

        logger.info(
            "SAGE Attention initialized: window=%d, sink=%d, top_k=%d, "
            "recent=%d, growing_recent_window=%s",
            self.sage_window_length,
            self.sage_num_sink_tokens,
            self.sage_top_k,
            self.sage_recent_window,
            self.sage_growing_recent_window,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SageAttentionMetadata:
        """Build SAGE attention metadata."""
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
        req_states: list[SageRequestState | None] = []
        is_long_list: list[bool] = []
        is_first_long_list: list[bool] = []

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        active_ids: set[str] = set()

        for i in range(num_reqs):
            rid = request_ids[i] if request_ids else ""
            if rid:
                active_ids.add(rid)
            seq_len_i = (
                int(seq_lens[i].item())
                if seq_lens.is_cpu
                else int(seq_lens.cpu()[i].item())
            )
            q_len_i = int(query_start_loc_cpu[i + 1] - query_start_loc_cpu[i])
            is_long = seq_len_i > self.sage_window_length
            state = self.state_registry.get_or_create(rid) if rid else None
            is_first = (
                is_long
                and q_len_i == 1
                and state is not None
                and not state.prefill_done
            )
            if is_first and state is not None:
                state.prefill_done = True
            req_states.append(state)
            is_long_list.append(is_long)
            is_first_long_list.append(is_first)

        # Prune finished requests.
        if active_ids:
            self.state_registry.prune(active_ids)

        # Keep these on CPU – they are only used for Python-level control flow
        # (branching / indexing), never for GPU computation.  Placing them on
        # GPU would cause illegal device-to-host syncs during CUDA graph
        # capture (`.any().item()` in forward()).
        is_long_context = torch.tensor(is_long_list, dtype=torch.bool, device="cpu")
        is_first_long_decode = torch.tensor(
            is_first_long_list, dtype=torch.bool, device="cpu"
        )

        return SageAttentionMetadata(
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
            is_long_context=is_long_context,
            is_first_long_decode=is_first_long_decode,
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repeat_kv(keys: torch.Tensor, values: torch.Tensor, n_rep: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand KV from kv_heads to query_heads by repeating.

    This is the standard GQA expansion: each KV head is repeated n_rep times
    to match the number of query heads.

    Args:
        keys: [seq_len, num_kv_heads, head_dim]
        values: [seq_len, num_kv_heads, head_dim]
        n_rep: number of times to repeat (num_query_heads // num_kv_heads)

    Returns:
        (keys, values) each [seq_len, num_query_heads, head_dim]
    """
    if n_rep == 1:
        return keys, values

    seq_len, num_kv_heads, head_dim = keys.shape
    # [seq_len, num_kv_heads, head_dim] -> [seq_len, num_kv_heads, 1, head_dim]
    keys = keys.unsqueeze(2)
    values = values.unsqueeze(2)
    # Expand and reshape: [seq_len, num_kv_heads, n_rep, head_dim]
    #                  -> [seq_len, num_kv_heads * n_rep, head_dim]
    keys = keys.expand(seq_len, num_kv_heads, n_rep, head_dim)
    values = values.expand(seq_len, num_kv_heads, n_rep, head_dim)
    keys = keys.reshape(seq_len, num_kv_heads * n_rep, head_dim)
    values = values.reshape(seq_len, num_kv_heads * n_rep, head_dim)
    return keys.contiguous(), values.contiguous()


def _gather_paged_kv(
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather KV pairs from the paged cache for a single request.

    Args:
        kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
        block_table_row: [max_blocks_per_seq] int32 block ids for this req.
        token_indices: 1-D int64 tensor of token positions to gather.
        block_size: tokens per block.

    Returns:
        (keys, values) each [len(token_indices), num_kv_heads, head_size]
    """
    key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache = kv_cache[1]

    block_ids = block_table_row[token_indices // block_size]  # physical blocks
    slot_offsets = token_indices % block_size

    # Gather: index [block_ids, slot_offsets] → [N, num_kv_heads, head_size]
    gathered_keys = key_cache[block_ids, slot_offsets]
    gathered_values = value_cache[block_ids, slot_offsets]
    return gathered_keys, gathered_values


def _topk_select_query_head_level(
    query_row: torch.Tensor,
    candidate_keys: torch.Tensor,
    candidate_values: torch.Tensor,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k K/V for one decode row at query head level.

    Following the sambaLLMs SAGE implementation, we do top-k selection at the
    query head level (not KV head level). Each query head independently selects
    its own top-k tokens. This is important for GQA models where different
    query heads may find different tokens important.

    Args:
        query_row: [num_query_heads, head_dim]
        candidate_keys: [num_candidates, num_query_heads, head_dim]
            (already expanded from KV heads)
        candidate_values: [num_candidates, num_query_heads, head_dim]
            (already expanded from KV heads)
        top_k: number of tokens to select per query head
        scale: softmax scale

    Returns:
        (topk_keys, topk_values) each [num_query_heads, top_k, head_dim]
    """
    num_candidates, num_query_heads, head_dim = candidate_keys.shape

    # query_row: [num_query_heads, head_dim] -> [num_query_heads, 1, head_dim]
    q = query_row.unsqueeze(1)

    # candidate_keys: [N, H_q, D] -> [H_q, N, D]
    k = candidate_keys.permute(1, 0, 2)

    # scores: [H_q, 1, N] = bmm(q, k.T)
    # q: [H_q, 1, D], k.T: [H_q, D, N] -> [H_q, 1, N]
    scores = torch.bmm(q, k.transpose(1, 2)) * scale
    scores = scores.squeeze(1)  # [H_q, N]

    actual_k = min(top_k, num_candidates)
    _, indices = torch.topk(scores, k=actual_k, dim=-1, sorted=False)
    # indices: [H_q, actual_k]

    # Gather top-k keys and values for each query head
    # indices: [H_q, actual_k] -> [H_q, actual_k, 1] for gather
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, head_dim)

    # Permute candidates to [H_q, N, D] for gather
    k_for_gather = candidate_keys.permute(1, 0, 2)  # [H_q, N, D]
    v_for_gather = candidate_values.permute(1, 0, 2)  # [H_q, N, D]

    # Gather: [H_q, actual_k, D]
    topk_keys = torch.gather(k_for_gather, dim=1, index=indices_expanded)
    topk_values = torch.gather(v_for_gather, dim=1, index=indices_expanded)

    return topk_keys, topk_values  # [num_query_heads, top_k, head_dim]


# ---------------------------------------------------------------------------
# Attention implementation
# ---------------------------------------------------------------------------


class SageAttentionImpl(AttentionImpl):
    """SAGE attention: standard FA for prefill/short-context decode,
    gather+topk+dense-FA for long-context decode."""

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
            raise ValueError("SAGE attention does not support ALiBi")

        self.sliding_window = (-1, -1)
        self.kv_cache_dtype = kv_cache_dtype

        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.attn_type = attn_type

        self.vllm_flash_attn_version = get_flash_attn_version(head_size=head_size)

        logger.info_once(
            "SAGE attention using FlashAttention version %s",
            self.vllm_flash_attn_version,
        )

        # SAGE config
        vllm_config = get_current_vllm_config()
        sage_cfg = vllm_config.sage_config
        if sage_cfg is not None:
            self.sage_window_length = sage_cfg.window_length
            self.sage_num_sink_tokens = sage_cfg.num_sink_tokens
            self.sage_top_k = sage_cfg.top_k
            self.sage_growing_recent_window = sage_cfg.growing_recent_window
        else:
            hf_config = vllm_config.model_config.hf_config
            self.sage_window_length = getattr(
                hf_config, "sage_window_length", SAGE_DEFAULT_WINDOW_LENGTH
            )
            self.sage_num_sink_tokens = getattr(
                hf_config, "sage_num_sink_tokens", SAGE_DEFAULT_NUM_SINK_TOKENS
            )
            self.sage_top_k = getattr(hf_config, "sage_top_k", SAGE_DEFAULT_TOP_K)
            self.sage_growing_recent_window = getattr(
                hf_config, "sage_growing_recent_window", False
            )

        self.sage_recent_window = (
            self.sage_window_length - self.sage_num_sink_tokens - self.sage_top_k
        )

        # Lazy-initialized cached tensors (set on first forward call when
        # device/dtype are known).  Avoids thousands of redundant tensor
        # allocations per decode step.
        self._cached_tensors_ready = False

    def _init_cached_tensors(self, device: torch.device,
                             dtype: torch.dtype) -> None:
        """One-time allocation of tensors that never change between steps."""
        num_sink = self.sage_num_sink_tokens
        # Sink token positions: always [0, 1, ..., num_sink-1]
        self._cached_sink_indices = torch.arange(
            num_sink, dtype=torch.int64, device=device)
        # cu_seqlens_q for a single-token query
        self._cached_cu_seqlens_q_one = torch.tensor(
            [0, 1], dtype=torch.int32, device=device)
        # Pre-compute sink block / slot decomposition
        block_size = None  # filled on first actual use (needs kv_cache shape)
        self._cached_sink_block_size = block_size
        self._cached_device = device
        self._cached_dtype = dtype
        self._cached_tensors_ready = True

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
        attn_metadata: SageAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        assert self.vllm_flash_attn_version is not None

        # Determine if there are any long-context decode rows.
        has_long_decode = (
            attn_metadata.is_long_context is not None
            and attn_metadata.max_query_len == 1
            and attn_metadata.is_long_context.any().item()
        )

        if not has_long_decode:
            # ---- Standard FlashAttention path (prefill, short decode) ----
            return self._flash_attn_forward(
                layer, query, kv_cache, attn_metadata, output
            )

        # ---- Mixed batch: some rows need SAGE, others are standard ----
        return self._sage_mixed_forward(layer, query, kv_cache, attn_metadata, output)

    # ---- Standard FA path ----

    def _flash_attn_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
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

    # ---- SAGE mixed path (batched) ----

    def _sage_mixed_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Handle a decode batch mixing standard and SAGE rows.

        Instead of iterating per-row (which causes O(num_reqs) GPU→CPU syncs
        and O(num_reqs × num_layers) kernel launches), this method partitions
        the batch into standard and SAGE groups and issues **one** FA kernel
        call per group.
        """
        assert attn_metadata.is_long_context is not None
        assert attn_metadata.request_states is not None
        device = query.device
        layer_idx = getattr(layer, "_layer_index", 0)

        # Lazy-init cached tensors on first call.
        if not self._cached_tensors_ready:
            self._init_cached_tensors(device, query.dtype)

        # --- Partition rows (CPU, no GPU sync) ---
        # is_long_context lives on CPU (see builder).
        sage_mask = attn_metadata.is_long_context  # [num_reqs] bool, CPU
        std_mask = ~sage_mask
        sage_row_cpu = sage_mask.nonzero(as_tuple=True)[0]  # CPU int64
        std_row_cpu = std_mask.nonzero(as_tuple=True)[0]

        N_sage = sage_row_cpu.shape[0]
        N_std = std_row_cpu.shape[0]

        # --- Standard (short-context) rows: one batched paged-FA call ---
        if N_std > 0:
            self._batched_standard_decode(
                layer, query, kv_cache, attn_metadata, output,
                std_row_cpu, N_std)

        # --- First-long-decode top-k selection (rare, per-row) ---
        if (attn_metadata.is_first_long_decode is not None
                and attn_metadata.is_first_long_decode.any()):
            block_size = kv_cache.shape[2]
            for idx in range(N_sage):
                row_i = int(sage_row_cpu[idx])
                if not attn_metadata.is_first_long_decode[row_i]:
                    continue
                self._compute_topk_for_row(
                    query, kv_cache, attn_metadata, row_i,
                    layer_idx, block_size, device)

        # --- SAGE (long-context) rows: batched gather + dense FA ---
        if N_sage > 0:
            self._batched_sage_decode(
                layer, query, kv_cache, attn_metadata, output,
                sage_row_cpu, N_sage, layer_idx)

        return output

    # ---- Batched standard decode ----

    def _batched_standard_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        output: torch.Tensor,
        std_row_cpu: torch.Tensor,
        N_std: int,
    ) -> None:
        """One FA kernel call for all short-context decode rows."""
        device = query.device
        key_cache, value_cache = kv_cache.unbind(0)

        std_gpu = std_row_cpu.to(device, dtype=torch.long, non_blocking=True)

        q_std = query[std_gpu]  # [N_std, num_heads, head_dim]
        seq_lens_std = attn_metadata.seq_lens[std_gpu]
        block_table_std = attn_metadata.block_table[std_gpu]

        cu_seqlens_q = torch.arange(
            0, N_std + 1, dtype=torch.int32, device=device)
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

    # ---- Top-k selection (once per request, at prefill→decode edge) ----

    def _compute_topk_for_row(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        row_i: int,
        layer_idx: int,
        block_size: int,
        device: torch.device,
    ) -> None:
        """Compute and store top-k KV for a single first-long-decode row."""
        state = attn_metadata.request_states[row_i]
        if state is None or layer_idx in state.topk_keys_per_layer:
            return

        seq_len = int(attn_metadata.seq_lens[row_i].item())
        block_table_row = attn_metadata.block_table[row_i]
        query_row = query[row_i]  # [num_heads, head_dim] (decode: q_len=1)

        num_sink = self.sage_num_sink_tokens
        recent = self.sage_recent_window
        top_k = self.sage_top_k

        # In growing mode, freeze the recent-window start at the first long
        # decode step. The first long decode is exactly the step at which
        # _compute_topk_for_row runs (gated by is_first_long_decode), so the
        # anchor is captured at most once per request.
        if (
            self.sage_growing_recent_window
            and state.recent_window_start is None
        ):
            state.recent_window_start = seq_len - recent

        cand_start = num_sink
        cand_end = seq_len - recent
        num_candidates = max(0, cand_end - cand_start)

        if num_candidates > 0:
            cand_indices = torch.arange(
                cand_start, cand_end, dtype=torch.int64, device=device)
            cand_keys, cand_values = _gather_paged_kv(
                kv_cache, block_table_row, cand_indices, block_size)
            cand_keys_exp, cand_values_exp = _repeat_kv(
                cand_keys, cand_values, self.num_queries_per_kv)
            topk_keys, topk_values = _topk_select_query_head_level(
                query_row, cand_keys_exp, cand_values_exp, top_k, self.scale)
            state.topk_keys_per_layer[layer_idx] = topk_keys
            state.topk_values_per_layer[layer_idx] = topk_values
        else:
            state.topk_keys_per_layer[layer_idx] = torch.empty(
                (self.num_heads, 0, self.head_size),
                device=device, dtype=query.dtype)
            state.topk_values_per_layer[layer_idx] = torch.empty(
                (self.num_heads, 0, self.head_size),
                device=device, dtype=query.dtype)

    # ---- Batched SAGE decode ----

    def _batched_sage_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        output: torch.Tensor,
        sage_row_cpu: torch.Tensor,
        N: int,
        layer_idx: int,
    ) -> None:
        """Batched SAGE decode for all long-context rows.

        Common preamble:
        1. One batched gather for sink tokens across all rows
        2. GQA expansion for sink
        3. One stack for stored top-k KV across all rows

        Recent gather + dense-KV assembly is mode-dependent:
        * Fixed recent window (StreamingLLM-style): uniform-length recent
          window per row, dense KV laid out as [N, wl, num_heads, D] and
          flattened.  See ``_assemble_dense_kv_fixed``.
        * Growing recent window: per-row variable length, dense KV packed
          into ``[total_dense, num_heads, D]`` with varlen ``cu_seqlens_k``.
          See ``_assemble_dense_kv_growing``.

        Common epilogue: one flash_attn_varlen_func call.
        """
        device = query.device
        num_sink = self.sage_num_sink_tokens
        top_k = self.sage_top_k
        head_dim = self.head_size
        block_size = kv_cache.shape[2]

        key_cache = kv_cache[0]    # [num_blocks, block_size, num_kv_heads, D]
        value_cache = kv_cache[1]

        sage_gpu = sage_row_cpu.to(device, dtype=torch.long, non_blocking=True)
        block_table_sage = attn_metadata.block_table[sage_gpu]  # [N, max_blk]
        seq_lens_sage = attn_metadata.seq_lens[sage_gpu]        # [N] GPU

        # ---- 1. Batched sink gather (uniform across rows) ----
        sink_idx = self._cached_sink_indices          # [num_sink]
        sink_blk = sink_idx // block_size             # [num_sink]
        sink_off = sink_idx % block_size              # [num_sink]

        sink_phys = block_table_sage[:, sink_blk]
        flat_sink_phys = sink_phys.reshape(-1)        # [N * num_sink]
        flat_sink_off = sink_off.unsqueeze(0).expand(N, -1).reshape(-1)

        sink_k = key_cache[flat_sink_phys, flat_sink_off].reshape(
            N, num_sink, self.num_kv_heads, head_dim)
        sink_v = value_cache[flat_sink_phys, flat_sink_off].reshape(
            N, num_sink, self.num_kv_heads, head_dim)

        # ---- 2. GQA expansion for sink (sink-only here; recent handled
        #         inside the mode-specific assembly) ----
        n_rep = self.num_queries_per_kv
        if n_rep > 1:
            sink_k = (sink_k.unsqueeze(3)
                      .expand(-1, -1, -1, n_rep, -1)
                      .reshape(N, num_sink, self.num_heads, head_dim))
            sink_v = (sink_v.unsqueeze(3)
                      .expand(-1, -1, -1, n_rep, -1)
                      .reshape(N, num_sink, self.num_heads, head_dim))

        # ---- 3. Stack stored top-k KV ----
        # Each state stores [num_q_heads, top_k, head_dim].
        topk_k_list: list[torch.Tensor] = []
        topk_v_list: list[torch.Tensor] = []
        for idx in range(N):
            row_i = int(sage_row_cpu[idx])
            state = attn_metadata.request_states[row_i]
            if state is not None and layer_idx in state.topk_keys_per_layer:
                # [num_q_heads, top_k, D] → [top_k, num_q_heads, D]
                topk_k_list.append(
                    state.topk_keys_per_layer[layer_idx].transpose(0, 1))
                topk_v_list.append(
                    state.topk_values_per_layer[layer_idx].transpose(0, 1))
            else:
                topk_k_list.append(torch.zeros(
                    top_k, self.num_heads, head_dim,
                    device=device, dtype=query.dtype))
                topk_v_list.append(torch.zeros(
                    top_k, self.num_heads, head_dim,
                    device=device, dtype=query.dtype))

        topk_k = torch.stack(topk_k_list, dim=0)  # [N, top_k, q_heads, D]
        topk_v = torch.stack(topk_v_list, dim=0)

        # ---- 4. Mode-specific recent gather + dense KV assembly ----
        if self.sage_growing_recent_window:
            dense_k, dense_v, cu_seqlens_k, max_seqlen_k = (
                self._assemble_dense_kv_growing(
                    key_cache, value_cache, block_table_sage, seq_lens_sage,
                    sink_k, sink_v, topk_k, topk_v,
                    sage_row_cpu, attn_metadata.request_states,
                    N, block_size, n_rep, query.dtype, device,
                )
            )
        else:
            dense_k, dense_v, cu_seqlens_k, max_seqlen_k = (
                self._assemble_dense_kv_fixed(
                    key_cache, value_cache, block_table_sage, seq_lens_sage,
                    sink_k, sink_v, topk_k, topk_v,
                    N, block_size, n_rep, device,
                )
            )

        # ---- 5. One batched dense FA call ----
        q_sage = query[sage_gpu]  # [N, num_q_heads, head_dim]
        cu_seqlens_q = torch.arange(
            0, N + 1, dtype=torch.int32, device=device)
        out_sage = torch.empty_like(q_sage)
        flash_attn_varlen_func(
            q=q_sage,
            k=dense_k,
            v=dense_v,
            out=out_sage,
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
        output[sage_gpu] = out_sage

    # ---- Dense-KV assembly: fixed-length recent window ----

    def _assemble_dense_kv_fixed(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table_sage: torch.Tensor,
        seq_lens_sage: torch.Tensor,
        sink_k: torch.Tensor,
        sink_v: torch.Tensor,
        topk_k: torch.Tensor,
        topk_v: torch.Tensor,
        N: int,
        block_size: int,
        n_rep: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Recent gather + dense KV assembly for the StreamingLLM-style
        fixed-length recent window.

        Each row's dense KV has the same length ``wl = num_sink + top_k +
        recent``, so we keep a [N, wl, num_heads, D] tensor and flatten it.
        ``cu_seqlens_k`` is the trivial ``arange * wl``.
        """
        num_sink = self.sage_num_sink_tokens
        recent = self.sage_recent_window
        head_dim = self.head_size

        # ---- Batched recent gather (uniform length = recent) ----
        offsets = torch.arange(recent, device=device, dtype=torch.long)
        # recent_pos[i, j] = seq_lens_sage[i] - recent + j
        recent_pos = (seq_lens_sage.unsqueeze(1).long() - recent
                      + offsets.unsqueeze(0))         # [N, recent]
        recent_blk_idx = recent_pos // block_size
        recent_slot_off = recent_pos % block_size

        recent_phys = torch.gather(
            block_table_sage.long(), 1, recent_blk_idx)
        flat_recent_phys = recent_phys.reshape(-1)
        flat_recent_off = recent_slot_off.reshape(-1)

        recent_k = key_cache[flat_recent_phys, flat_recent_off].reshape(
            N, recent, self.num_kv_heads, head_dim)
        recent_v = value_cache[flat_recent_phys, flat_recent_off].reshape(
            N, recent, self.num_kv_heads, head_dim)

        if n_rep > 1:
            recent_k = (recent_k.unsqueeze(3)
                        .expand(-1, -1, -1, n_rep, -1)
                        .reshape(N, recent, self.num_heads, head_dim))
            recent_v = (recent_v.unsqueeze(3)
                        .expand(-1, -1, -1, n_rep, -1)
                        .reshape(N, recent, self.num_heads, head_dim))

        # ---- Concatenate dense KV ----
        dense_k = torch.cat([sink_k, topk_k, recent_k], dim=1)
        dense_v = torch.cat([sink_v, topk_v, recent_v], dim=1)
        wl = dense_k.shape[1]

        dense_k = dense_k.reshape(N * wl, self.num_heads, head_dim)
        dense_v = dense_v.reshape(N * wl, self.num_heads, head_dim)

        cu_seqlens_k = (torch.arange(
            0, N + 1, dtype=torch.int32, device=device) * wl)
        max_seqlen_k = wl

        return dense_k, dense_v, cu_seqlens_k, max_seqlen_k

    # ---- Dense-KV assembly: growing recent window ----

    def _assemble_dense_kv_growing(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table_sage: torch.Tensor,
        seq_lens_sage: torch.Tensor,
        sink_k: torch.Tensor,
        sink_v: torch.Tensor,
        topk_k: torch.Tensor,
        topk_v: torch.Tensor,
        sage_row_cpu: torch.Tensor,
        request_states: list,
        N: int,
        block_size: int,
        n_rep: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Recent gather + dense KV assembly for the growing recent window.

        Each row has a different recent length ``r_i = seq_len_i -
        recent_window_start_i``, so the dense KV is packed varlen-style:
        ``dense_k`` shape is ``[total_dense, num_heads, head_dim]`` with
        ``cu_seqlens_k[i+1] - cu_seqlens_k[i] = num_sink + top_k + r_i``.
        """
        num_sink = self.sage_num_sink_tokens
        top_k = self.sage_top_k
        head_dim = self.head_size
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads

        # ---- 1. Per-row recent_window_start (frozen anchor) ----
        # Built on CPU then moved to GPU once: anchors live as Python ints
        # on SageRequestState.  N is small (number of long-context decode
        # rows in this batch), so this loop is cheap.
        recent_starts_cpu = torch.empty(N, dtype=torch.long)
        for idx in range(N):
            row_i = int(sage_row_cpu[idx])
            state = request_states[row_i]
            if state is not None and state.recent_window_start is not None:
                recent_starts_cpu[idx] = state.recent_window_start
            else:
                # Defensive fallback: should not happen because
                # _compute_topk_for_row sets the anchor at first long decode.
                # Fall back to the streaming-style start so attention is
                # still well-defined.
                recent_starts_cpu[idx] = -self.sage_recent_window
        recent_starts = recent_starts_cpu.to(
            device, dtype=torch.long, non_blocking=True)
        # If we fell back (recent_starts_cpu < 0), interpret it relative to
        # the current seq_len: -recent → seq_len - recent.
        recent_starts = torch.where(
            recent_starts < 0,
            seq_lens_sage.long() + recent_starts,
            recent_starts,
        )

        # ---- 2. Per-row recent lengths and packed cu_seqlens_k ----
        r = seq_lens_sage.long() - recent_starts       # [N]
        wl = (num_sink + top_k) + r                    # [N]
        cu_seqlens_k = torch.empty(
            N + 1, dtype=torch.int32, device=device)
        cu_seqlens_k[0] = 0
        cu_seqlens_k[1:] = wl.to(torch.int32).cumsum(0, dtype=torch.int32)
        dense_start = cu_seqlens_k[:-1].long()         # [N]

        # Single GPU→CPU sync to learn total_dense + max_seqlen_k.
        totals = torch.stack([wl.sum(), wl.max()]).cpu()
        total_dense = int(totals[0].item())
        max_seqlen_k = int(totals[1].item())
        total_recent = total_dense - N * (num_sink + top_k)

        # ---- 3. Allocate packed dense buffer ----
        dense_k = torch.empty(
            total_dense, num_heads, head_dim, device=device, dtype=dtype)
        dense_v = torch.empty_like(dense_k)

        # ---- 4. Scatter sink (uniform per-row width) ----
        arange_sink = torch.arange(
            num_sink, device=device, dtype=torch.long)
        sink_dest = (dense_start.unsqueeze(1)
                     + arange_sink.unsqueeze(0)).reshape(-1)  # [N*num_sink]
        dense_k[sink_dest] = sink_k.reshape(-1, num_heads, head_dim)
        dense_v[sink_dest] = sink_v.reshape(-1, num_heads, head_dim)

        # ---- 5. Scatter topk (uniform per-row width) ----
        arange_topk = torch.arange(
            top_k, device=device, dtype=torch.long)
        topk_dest = (dense_start.unsqueeze(1) + num_sink
                     + arange_topk.unsqueeze(0)).reshape(-1)  # [N*top_k]
        dense_k[topk_dest] = topk_k.reshape(-1, num_heads, head_dim)
        dense_v[topk_dest] = topk_v.reshape(-1, num_heads, head_dim)

        # ---- 6. Gather + scatter recent (variable per-row width) ----
        # row_id_flat[k] = row index of the k-th element across the packed
        # recent layout (length total_recent).
        row_id_flat = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            r,
            output_size=total_recent,
        )

        # Per-row starting offsets within the packed recent layout.
        recent_cumsum = torch.empty(
            N + 1, dtype=torch.long, device=device)
        recent_cumsum[0] = 0
        recent_cumsum[1:] = r.cumsum(0)
        flat_arange = torch.arange(
            total_recent, device=device, dtype=torch.long)
        offset_in_row = flat_arange - recent_cumsum[row_id_flat]

        # Source token positions in each request's sequence.
        recent_positions = recent_starts[row_id_flat] + offset_in_row
        recent_blk = recent_positions // block_size
        recent_off = recent_positions % block_size
        recent_phys = block_table_sage.long()[row_id_flat, recent_blk]

        recent_k_kv = key_cache[recent_phys, recent_off]
        recent_v_kv = value_cache[recent_phys, recent_off]

        if n_rep > 1:
            recent_k_pkd = (recent_k_kv.unsqueeze(2)
                            .expand(-1, num_kv_heads, n_rep, head_dim)
                            .reshape(total_recent, num_heads, head_dim))
            recent_v_pkd = (recent_v_kv.unsqueeze(2)
                            .expand(-1, num_kv_heads, n_rep, head_dim)
                            .reshape(total_recent, num_heads, head_dim))
        else:
            recent_k_pkd = recent_k_kv
            recent_v_pkd = recent_v_kv

        recent_dest = (dense_start[row_id_flat] + num_sink + top_k
                       + offset_in_row)
        dense_k[recent_dest] = recent_k_pkd
        dense_v[recent_dest] = recent_v_pkd

        return dense_k, dense_v, cu_seqlens_k, max_seqlen_k
