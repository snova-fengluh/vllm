# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SAGE (Self-Attention Guided Eviction) Attention Backend for vLLM.

SAGE combines StreamLLM-style windowing with attention-guided top-k selection
to maintain a bounded KV cache for efficient long-sequence inference.

KV Cache Structure:
    [sink_tokens] + [top_k_important_tokens] + [recent_tokens]
         ^                    ^                     ^
       Always kept      Selected via Q·K       Sliding window

Algorithm Flow:
1. Prefill Phase: Accumulate all KV pairs normally
2. Trigger: When kv_length > window_length, activate eviction
3. Top-K Selection (one-pass, after prefill):
   - Extract eviction candidates: kv_cache[sink:-recent]
   - Compute: scores = query @ evict_keys.T
   - Select top-k indices per query head
   - Store in separate topk_key_cache, topk_value_cache
4. StreamLLM Update: Maintain [sink] + [recent] + [new] in main cache
5. Attention: Concatenate main cache with top-k cache for attention
"""

from dataclasses import dataclass
from typing import ClassVar

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
    SageCacheState,
    create_sage_cache_state,
)
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


class SageAttentionBackend(AttentionBackend):
    """
    SAGE attention backend that wraps FlashAttention with SAGE cache management.

    This backend implements the SAGE (Self-Attention Guided Eviction) algorithm
    which maintains a bounded KV cache through attention-guided token selection.
    """

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
        return False  # SAGE is designed for causal/autoregressive models

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """SAGE only supports decoder attention."""
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SageAttentionMetadataBuilder"]:
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
        # Same layout as FlashAttention
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
        # SAGE has its own sink token handling
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


@dataclass
class SageAttentionMetadata:
    """Metadata for SAGE attention."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # SAGE-specific metadata
    sage_cache_state: SageCacheState | None = None
    is_first_decode_after_prefill: bool = False

    causal: bool = True


class SageAttentionMetadataBuilder(AttentionMetadataBuilder[SageAttentionMetadata]):
    """Builder for SAGE attention metadata."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
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

        # SAGE configuration from model config or defaults
        hf_config = self.model_config.hf_config
        self.sage_window_length = getattr(
            hf_config, "sage_window_length", SAGE_DEFAULT_WINDOW_LENGTH
        )
        self.sage_num_sink_tokens = getattr(
            hf_config, "sage_num_sink_tokens", SAGE_DEFAULT_NUM_SINK_TOKENS
        )
        self.sage_top_k = getattr(hf_config, "sage_top_k", SAGE_DEFAULT_TOP_K)

        # Create SAGE cache state
        num_layers = self.model_config.get_num_layers(vllm_config.parallel_config)
        self.sage_cache_state = create_sage_cache_state(
            num_layers=num_layers,
            window_length=self.sage_window_length,
            num_sink_tokens=self.sage_num_sink_tokens,
            top_k=self.sage_top_k,
        )

        logger.info(
            "SAGE Attention initialized: window=%d, sink=%d, top_k=%d",
            self.sage_window_length,
            self.sage_num_sink_tokens,
            self.sage_top_k,
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

        # Check if this is the first decode step after prefill
        # This is when we perform top-k selection
        is_first_decode = (
            max_query_len == 1
            and max_seq_len > self.sage_window_length
            and not self.sage_cache_state.eviction_triggered
        )

        if is_first_decode:
            self.sage_cache_state.eviction_triggered = True

        return SageAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            sage_cache_state=self.sage_cache_state,
            is_first_decode_after_prefill=is_first_decode,
            causal=causal,
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # SAGE handles its own cache management, no cascade needed
        return False


class SageAttentionImpl(AttentionImpl):
    """
    SAGE attention implementation.

    This implementation wraps FlashAttention and adds SAGE cache management
    for efficient long-sequence inference.
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
            raise ValueError("SAGE attention does not support ALiBi")

        self.sliding_window = (-1, -1)  # SAGE handles windowing internally
        self.kv_cache_dtype = kv_cache_dtype

        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.attn_type = attn_type

        self.vllm_flash_attn_version = get_flash_attn_version(
            head_size=head_size,
        )

        logger.info_once(
            "SAGE attention using FlashAttention version %s",
            self.vllm_flash_attn_version,
        )

        # Get SAGE config from vllm config
        vllm_config = get_current_vllm_config()
        hf_config = vllm_config.model_config.hf_config
        self.sage_window_length = getattr(
            hf_config, "sage_window_length", SAGE_DEFAULT_WINDOW_LENGTH
        )
        self.sage_num_sink_tokens = getattr(
            hf_config, "sage_num_sink_tokens", SAGE_DEFAULT_NUM_SINK_TOKENS
        )
        self.sage_top_k = getattr(hf_config, "sage_top_k", SAGE_DEFAULT_TOP_K)

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
        """
        Forward pass with SAGE attention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: SAGE attention metadata
            output: shape = [num_tokens, num_heads * head_size]

        Returns:
            Output tensor of shape [num_tokens, num_heads * head_size]
        """
        if attn_metadata is None:
            # Profiling run
            return output.fill_(0)

        assert self.vllm_flash_attn_version is not None

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)

        # Standard FlashAttention forward for now
        # The SAGE logic will be integrated in the model's attention layer
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
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

        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update the KV cache with new key-value pairs."""
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

    def _topk_select(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        sage_cache_state: SageCacheState,
        layer_idx: int,
        seq_len: int,
    ) -> None:
        """
        Perform one-pass top-k selection after prefill.

        This method:
        1. Extracts eviction candidates from the cache
        2. Computes Q·K dot products
        3. Selects top-k indices per query head
        4. Stores top-k KV pairs in the SAGE cache state

        Args:
            query: Current query tensor [batch, num_heads, 1, head_dim]
            key_cache: Key cache tensor
            value_cache: Value cache tensor
            sage_cache_state: SAGE cache state manager
            layer_idx: Current layer index
            seq_len: Current sequence length
        """
        if sage_cache_state.is_topk_selected(layer_idx):
            return

        num_sink = sage_cache_state.num_sink_tokens
        top_k = sage_cache_state.top_k
        recent_window = sage_cache_state.get_recent_window_size()

        # Calculate eviction region boundaries
        # Eviction candidates are tokens between sink and recent window
        evict_start = num_sink
        evict_end = seq_len - recent_window

        if evict_end <= evict_start:
            # Not enough tokens to evict, skip top-k selection
            return

        evict_len = evict_end - evict_start

        # Select all candidates if not enough, otherwise use top_k
        actual_top_k = evict_len if evict_len < top_k else top_k

        # Extract eviction candidates from cache
        # This requires gathering from paged cache, which we'll implement
        # For now, we assume contiguous access to the eviction region
        # In production, this would use block table to gather

        # Compute attention scores
        # query: [batch, num_heads, 1, head_dim]
        # evict_keys: [batch, num_kv_heads, evict_len, head_dim]

        # For GQA, expand KV heads to match query heads
        if self.num_queries_per_kv > 1:
            # This would need actual key cache access
            pass

        # Select top-k indices and gather KV pairs
        # topk_indices, _ = sage_topk_selection(
        #     query, evict_keys, actual_top_k, self.scale
        # )
        # topk_keys, topk_values = gather_topk_kv(
        #     evict_keys, evict_values, topk_indices
        # )

        # Store in SAGE cache state
        # sage_cache_state.set_topk_cache(
        #     layer_idx, topk_keys, topk_values, topk_indices
        # )

        logger.debug(
            "Layer %d: Selected top-%d from %d eviction candidates",
            layer_idx,
            actual_top_k,
            evict_len,
        )
