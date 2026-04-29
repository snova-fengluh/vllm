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
                             paged[seq_len - (recent - 1) : seq_len])
  4. Run flash_attn_varlen_func against the dense KV (causal=False).
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
        else:
            hf_config = self.model_config.hf_config
            self.sage_window_length = getattr(
                hf_config, "sage_window_length", SAGE_DEFAULT_WINDOW_LENGTH
            )
            self.sage_num_sink_tokens = getattr(
                hf_config, "sage_num_sink_tokens", SAGE_DEFAULT_NUM_SINK_TOKENS
            )
            self.sage_top_k = getattr(hf_config, "sage_top_k", SAGE_DEFAULT_TOP_K)

        self.sage_recent_window = (
            self.sage_window_length - self.sage_num_sink_tokens - self.sage_top_k
        )

        # Per-request state registry
        self.state_registry = SageRequestStateRegistry()

        logger.info(
            "SAGE Attention initialized: window=%d, sink=%d, top_k=%d, recent=%d",
            self.sage_window_length,
            self.sage_num_sink_tokens,
            self.sage_top_k,
            self.sage_recent_window,
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

        # Debug logging
        num_long = sum(is_long_list)
        num_first_long = sum(is_first_long_list)

        # Get actual seq_len values for debugging
        seq_lens_list = []
        for i in range(min(num_reqs, 5)):  # Show first 5
            seq_len_i = (
                int(seq_lens[i].item())
                if seq_lens.is_cpu
                else int(seq_lens.cpu()[i].item())
            )
            seq_lens_list.append(seq_len_i)

        logger.info(
            "[SAGE-BUILD] num_reqs=%d, max_query_len=%d, max_seq_len=%d, "
            "num_long_context=%d, num_first_long_decode=%d, "
            "sage_window_length=%d, first_seq_lens=%s",
            num_reqs,
            max_query_len,
            max_seq_len,
            num_long,
            num_first_long,
            self.sage_window_length,
            seq_lens_list,
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
        else:
            hf_config = vllm_config.model_config.hf_config
            self.sage_window_length = getattr(
                hf_config, "sage_window_length", SAGE_DEFAULT_WINDOW_LENGTH
            )
            self.sage_num_sink_tokens = getattr(
                hf_config, "sage_num_sink_tokens", SAGE_DEFAULT_NUM_SINK_TOKENS
            )
            self.sage_top_k = getattr(hf_config, "sage_top_k", SAGE_DEFAULT_TOP_K)

        self.sage_recent_window = (
            self.sage_window_length - self.sage_num_sink_tokens - self.sage_top_k
        )

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

        # Get layer index for conditional logging (only log on layer 0)
        layer_idx = getattr(layer, "_layer_index", 0)

        if not has_long_decode:
            # ---- Standard FlashAttention path (prefill, short decode) ----
            if layer_idx == 0:
                # Determine why SAGE path was not taken
                if attn_metadata.max_query_len > 1:
                    reason = "prefill"
                else:
                    reason = "short_context"
                if attn_metadata.is_long_context is not None:
                    any_long = attn_metadata.is_long_context.any().item()
                else:
                    any_long = False
                logger.info(
                    "[SAGE-FORWARD] Standard FA path (reason=%s): "
                    "query.shape=%s, max_query_len=%d, max_seq_len=%d, "
                    "any_long_context=%s, window_length=%d",
                    reason,
                    tuple(query.shape),
                    attn_metadata.max_query_len,
                    attn_metadata.max_seq_len,
                    any_long,
                    self.sage_window_length,
                )
            return self._flash_attn_forward(
                layer, query, kv_cache, attn_metadata, output
            )

        # ---- Mixed batch: some rows need SAGE, others are standard ----
        if layer_idx == 0:
            num_long = attn_metadata.is_long_context.sum().item()
            logger.info(
                "[SAGE-FORWARD] SAGE mixed path: query.shape=%s, "
                "num_reqs=%d, num_long_context=%d",
                tuple(query.shape),
                attn_metadata.seq_lens.shape[0],
                num_long,
            )
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

    # ---- SAGE mixed path ----

    def _sage_mixed_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Handle a batch mixing standard and SAGE rows."""
        num_reqs = attn_metadata.seq_lens.shape[0]
        assert attn_metadata.is_long_context is not None
        assert attn_metadata.request_states is not None
        query_start_loc = attn_metadata.query_start_loc

        # Detect the layer index from the layer's prefix (set by Attention).
        layer_idx = getattr(layer, "_layer_index", 0)

        # Process each row.  For a decode-only batch every row has q_len = 1,
        # so query[i] is the i-th row's single query token.
        for i in range(num_reqs):
            q_start = int(query_start_loc[i].item())
            q_end = int(query_start_loc[i + 1].item())
            q_len = q_end - q_start

            if q_len != 1 or not attn_metadata.is_long_context[i]:
                # Prefill or short-context decode → standard FA for this row.
                self._standard_row(
                    layer,
                    query,
                    kv_cache,
                    attn_metadata,
                    output,
                    row_idx=i,
                    q_start=q_start,
                    q_end=q_end,
                )
                continue

            # Long-context decode → SAGE path.
            state = attn_metadata.request_states[i]
            seq_len = int(attn_metadata.seq_lens[i].item())
            block_table_row = attn_metadata.block_table[i]

            self._sage_decode_one_row(
                layer=layer,
                query_row=query[q_start],  # [num_heads, head_dim]
                kv_cache=kv_cache,
                block_table_row=block_table_row,
                seq_len=seq_len,
                state=state,
                layer_idx=layer_idx,
                is_first_long=(
                    attn_metadata.is_first_long_decode is not None
                    and attn_metadata.is_first_long_decode[i].item()
                ),
                output_row=output[q_start],  # [num_heads, head_dim]
            )

        return output

    def _standard_row(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        output: torch.Tensor,
        row_idx: int,
        q_start: int,
        q_end: int,
    ) -> None:
        """Run standard FlashAttention for a single row (slice of batch)."""
        key_cache, value_cache = kv_cache.unbind(0)
        q_len = q_end - q_start

        cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=query.device)
        seq_used_k = attn_metadata.seq_lens[row_idx : row_idx + 1]
        block_table = attn_metadata.block_table[row_idx : row_idx + 1]

        descale_shape = (1, self.num_kv_heads)
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        flash_attn_varlen_func(
            q=query[q_start:q_end],
            k=key_cache,
            v=value_cache,
            out=output[q_start:q_end],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=q_len,
            seqused_k=seq_used_k,
            max_seqlen_k=int(seq_used_k.item()),
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=None,
            window_size=list(self.sliding_window),
            block_table=block_table,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    def _sage_decode_one_row(
        self,
        layer: torch.nn.Module,
        query_row: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        seq_len: int,
        state: SageRequestState | None,
        layer_idx: int,
        is_first_long: bool,
        output_row: torch.Tensor,
    ) -> None:
        """Run SAGE decode for a single long-context row.

        Following the sambaLLMs SAGE implementation:
        1. Top-k selection is done at QUERY HEAD level (not KV head level)
        2. Each query head independently selects its top-k important tokens
        3. Top-k K/V tensors are stored at query head level
        4. Final dense K/V is at query head level with exactly window_length tokens

        This ensures that:
        - Different query heads in the same GQA group can select different tokens
        - The total K/V size is always exactly window_length (sink + top_k + recent)
        """
        num_sink = self.sage_num_sink_tokens
        recent = self.sage_recent_window
        top_k = self.sage_top_k
        block_size = kv_cache.shape[2]  # [2, num_blocks, block_size, ...]
        device = query_row.device

        # Only log on layer 0 to avoid redundant output across all layers
        if layer_idx == 0:
            logger.info(
                "[SAGE-DECODE-ROW] seq_len=%d, is_first_long=%s, "
                "query_row.shape=%s, block_table_row.shape=%s, "
                "num_query_heads=%d, num_kv_heads=%d",
                seq_len,
                is_first_long,
                tuple(query_row.shape),
                tuple(block_table_row.shape),
                self.num_heads,
                self.num_kv_heads,
            )

        # --- Top-k selection (once per request at prefill→decode edge) ---
        # Following sambaLLMs: select top-k at query head level
        if (
            is_first_long
            and state is not None
            and layer_idx not in state.topk_keys_per_layer
        ):
            # Gather candidate KV: tokens [num_sink .. seq_len - recent]
            cand_start = num_sink
            cand_end = seq_len - recent
            num_candidates = max(0, cand_end - cand_start)

            if layer_idx == 0:
                logger.info(
                    "[SAGE-TOPK-SELECT] cand_range=[%d:%d], num_candidates=%d, "
                    "top_k=%d, selecting at query head level",
                    cand_start,
                    cand_end,
                    num_candidates,
                    top_k,
                )

            if num_candidates > 0:
                cand_indices = torch.arange(
                    cand_start, cand_end, dtype=torch.int64, device=device
                )
                # Gather at KV head level: [num_candidates, num_kv_heads, head_dim]
                cand_keys, cand_values = _gather_paged_kv(
                    kv_cache, block_table_row, cand_indices, block_size
                )

                # Expand to query head level for top-k selection
                # [num_candidates, num_kv_heads, dim] -> [num_candidates, num_q_heads, dim]
                cand_keys_expanded, cand_values_expanded = _repeat_kv(
                    cand_keys, cand_values, self.num_queries_per_kv
                )

                # Select top-k at query head level
                # Returns: [num_query_heads, top_k, head_dim]
                topk_keys, topk_values = _topk_select_query_head_level(
                    query_row,
                    cand_keys_expanded,
                    cand_values_expanded,
                    top_k,
                    self.scale,
                )

                # Store at query head level
                state.topk_keys_per_layer[layer_idx] = topk_keys
                state.topk_values_per_layer[layer_idx] = topk_values

                if layer_idx == 0:
                    logger.info(
                        "[SAGE-TOPK-SELECT] cand_keys.shape=%s, "
                        "cand_keys_expanded.shape=%s, "
                        "topk_keys.shape=%s (per query head)",
                        tuple(cand_keys.shape),
                        tuple(cand_keys_expanded.shape),
                        tuple(topk_keys.shape),
                    )
            else:
                # No candidates to select from (edge case)
                head_dim = self.head_size
                state.topk_keys_per_layer[layer_idx] = torch.empty(
                    (self.num_heads, 0, head_dim), device=device, dtype=query_row.dtype
                )
                state.topk_values_per_layer[layer_idx] = torch.empty(
                    (self.num_heads, 0, head_dim), device=device, dtype=query_row.dtype
                )

        # --- Build dense KV at query head level ---
        # Structure: [sink_tokens] + [topk_tokens] + [recent_tokens]
        # All at query head level: [window_length, num_query_heads, head_dim]

        # 1. Gather sink tokens at KV head level, then expand
        sink_indices = torch.arange(0, num_sink, dtype=torch.int64, device=device)
        sink_keys, sink_values = _gather_paged_kv(
            kv_cache, block_table_row, sink_indices, block_size
        )
        # Expand to query heads: [num_sink, num_query_heads, head_dim]
        sink_keys, sink_values = _repeat_kv(
            sink_keys, sink_values, self.num_queries_per_kv
        )

        # 2. Gather recent tokens at KV head level, then expand
        recent_start = max(seq_len - recent, num_sink)
        recent_indices = torch.arange(
            recent_start, seq_len, dtype=torch.int64, device=device
        )
        recent_keys, recent_values = _gather_paged_kv(
            kv_cache, block_table_row, recent_indices, block_size
        )
        # Expand to query heads: [num_recent, num_query_heads, head_dim]
        recent_keys, recent_values = _repeat_kv(
            recent_keys, recent_values, self.num_queries_per_kv
        )

        # 3. Get stored top-k K/V (already at query head level)
        if state is not None and layer_idx in state.topk_keys_per_layer:
            # topk: [num_query_heads, top_k, head_dim]
            topk_keys = state.topk_keys_per_layer[layer_idx]
            topk_values = state.topk_values_per_layer[layer_idx]
            # Transpose to [top_k, num_query_heads, head_dim] for concatenation
            topk_keys = topk_keys.transpose(0, 1)
            topk_values = topk_values.transpose(0, 1)
        else:
            # No top-k available (shouldn't happen in normal flow)
            topk_keys = torch.empty(
                (0, self.num_heads, self.head_size),
                device=device,
                dtype=query_row.dtype,
            )
            topk_values = torch.empty(
                (0, self.num_heads, self.head_size),
                device=device,
                dtype=query_row.dtype,
            )

        # 4. Concatenate: [sink] + [topk] + [recent] at query head level
        # Each tensor is [seq_portion, num_query_heads, head_dim]
        dense_keys = torch.cat([sink_keys, topk_keys, recent_keys], dim=0)
        dense_values = torch.cat([sink_values, topk_values, recent_values], dim=0)

        window_len = dense_keys.shape[0]

        if layer_idx == 0:
            logger.info(
                "[SAGE-DENSE-KV] sink=%d, topk=%d, recent=%d, "
                "window_len=%d (should equal %d), dense_keys.shape=%s",
                len(sink_indices),
                topk_keys.shape[0],
                len(recent_indices),
                window_len,
                self.sage_window_length,
                tuple(dense_keys.shape),
            )

        # Run FlashAttention on the dense KV (single-sequence, non-paged).
        # K/V are at query head level, so this is MHA (not GQA).
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, window_len], dtype=torch.int32, device=device)

        # query_row: [num_heads, head_dim] → [1, num_heads, head_dim]
        q = query_row.unsqueeze(0)

        # K/V: [window_len, num_query_heads, head_dim] - same head count as Q
        out_buf = output_row.unsqueeze(0)  # [1, num_heads, head_dim]

        flash_attn_varlen_func(
            q=q,
            k=dense_keys,
            v=dense_values,
            out=out_buf,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=window_len,
            softmax_scale=self.scale,
            causal=False,  # not causal – positions are non-contiguous
            alibi_slopes=None,
            window_size=[-1, -1],
            block_table=None,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
        )

        if layer_idx == 0:
            logger.info(
                "[SAGE-DECODE-DONE] Completed SAGE decode: q.shape=%s, "
                "k.shape=%s, window_len=%d (logged for layer 0 only)",
                tuple(q.shape),
                tuple(dense_keys.shape),
                window_len,
            )
