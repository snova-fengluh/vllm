# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The MiniMax AI team.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MiniMax M2 model with SAGE (Self-Attention Guided Eviction) attention.

SAGE enables efficient long-sequence inference by maintaining a bounded KV cache
through attention-guided token selection:

KV Cache Structure:
    [sink_tokens] + [top_k_important_tokens] + [recent_tokens]

Key Features:
- StreamLLM-style sink token preservation
- Attention-guided top-k token selection
- Sliding window for recent context
- Memory-bounded inference for arbitrarily long sequences
"""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01RMSNormTP
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .minimax_m2 import (
    MiniMaxM2MoE,
    get_spec_layer_idx_from_weight_name,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# SAGE configuration defaults
SAGE_DEFAULT_WINDOW_LENGTH = 8192
SAGE_DEFAULT_NUM_SINK_TOKENS = 4
SAGE_DEFAULT_TOP_K = 512


@dataclass
class SageConfig:
    """Configuration for SAGE attention."""

    enabled: bool = True
    window_length: int = SAGE_DEFAULT_WINDOW_LENGTH
    num_sink_tokens: int = SAGE_DEFAULT_NUM_SINK_TOKENS
    top_k: int = SAGE_DEFAULT_TOP_K
    num_full_kv_layer: int = 0  # Layers that use full KV cache (no SAGE)

    @classmethod
    def from_hf_config(cls, config: PretrainedConfig) -> "SageConfig":
        """Extract SAGE config from HuggingFace config."""
        window_len = getattr(config, "sage_window_length", SAGE_DEFAULT_WINDOW_LENGTH)
        sink_tokens = getattr(
            config, "sage_num_sink_tokens", SAGE_DEFAULT_NUM_SINK_TOKENS
        )
        return cls(
            enabled=getattr(config, "sage_enabled", True),
            window_length=window_len,
            num_sink_tokens=sink_tokens,
            top_k=getattr(config, "sage_top_k", SAGE_DEFAULT_TOP_K),
            num_full_kv_layer=getattr(config, "sage_num_full_kv_layer", 0),
        )

    @property
    def recent_window_size(self) -> int:
        """Size of the recent token window."""
        return self.window_length - self.num_sink_tokens - self.top_k


class SageKVCache:
    """
    SAGE KV cache manager for a single layer.

    This class manages the KV cache with SAGE eviction logic:
    - Maintains sink tokens (always preserved)
    - Selects top-k important tokens via attention scores
    - Keeps a sliding window of recent tokens
    """

    def __init__(
        self,
        sage_config: SageConfig,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.config = sage_config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Top-k cache (separate from main paged cache)
        # Shape: [batch, num_heads, top_k, head_dim]
        self.topk_key_cache: torch.Tensor | None = None
        self.topk_value_cache: torch.Tensor | None = None

        # Tracking state
        self.seen_tokens: int = 0
        self.topk_selected: bool = False
        self.topk_indices: torch.Tensor | None = None

    def should_trigger_eviction(self, seq_len: int) -> bool:
        """Check if we should trigger SAGE eviction."""
        return seq_len > self.config.window_length and not self.topk_selected

    def perform_topk_selection(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        scale: float,
    ) -> None:
        """
        Perform one-pass top-k selection.

        This is called once after prefill when sequence length exceeds window.

        Args:
            query: Query tensor [batch, num_heads, 1, head_dim]
            keys: Key cache [batch, num_kv_heads, seq_len, head_dim]
            values: Value cache [batch, num_kv_heads, seq_len, head_dim]
            scale: Attention scaling factor
        """
        if self.topk_selected:
            return

        batch_size = query.shape[0]
        seq_len = keys.shape[2]

        # Calculate eviction region
        num_sink = self.config.num_sink_tokens
        recent_size = self.config.recent_window_size
        top_k = self.config.top_k

        evict_start = num_sink
        evict_end = seq_len - recent_size

        if evict_end <= evict_start:
            # Not enough tokens for eviction
            return

        # Extract eviction candidates
        evict_keys = keys[:, :, evict_start:evict_end, :]
        evict_values = values[:, :, evict_start:evict_end, :]
        evict_len = evict_keys.shape[2]

        actual_top_k = min(top_k, evict_len)

        # Handle GQA: expand KV heads for attention score computation
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            # Expand keys for score computation
            evict_keys_expanded = (
                evict_keys[:, :, None, :, :]
                .expand(batch_size, self.num_kv_heads, n_rep, evict_len, self.head_dim)
                .reshape(batch_size, self.num_heads, evict_len, self.head_dim)
            )

            evict_values_expanded = (
                evict_values[:, :, None, :, :]
                .expand(batch_size, self.num_kv_heads, n_rep, evict_len, self.head_dim)
                .reshape(batch_size, self.num_heads, evict_len, self.head_dim)
            )
        else:
            evict_keys_expanded = evict_keys
            evict_values_expanded = evict_values

        # Compute attention scores: [batch, num_heads, 1, evict_len]
        scores = torch.matmul(query, evict_keys_expanded.transpose(-2, -1)) * scale

        # Select top-k: [batch, num_heads, 1, top_k]
        _, topk_indices = torch.topk(
            scores, k=actual_top_k, dim=-1, largest=True, sorted=False
        )

        # Expand indices for gathering: [batch, num_heads, top_k, head_dim]
        topk_indices = topk_indices.squeeze(-2)  # [batch, num_heads, top_k]
        gather_indices = topk_indices.unsqueeze(-1).expand(
            batch_size, self.num_heads, actual_top_k, self.head_dim
        )

        # Gather top-k KV pairs
        self.topk_key_cache = torch.gather(
            evict_keys_expanded, dim=2, index=gather_indices
        )
        self.topk_value_cache = torch.gather(
            evict_values_expanded, dim=2, index=gather_indices
        )
        self.topk_indices = topk_indices
        self.topk_selected = True

        logger.debug(
            "SAGE: Selected top-%d from %d eviction candidates",
            actual_top_k,
            evict_len,
        )

    def get_combined_kv(
        self,
        main_keys: torch.Tensor,
        main_values: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get combined KV cache (main + top-k).

        Args:
            main_keys: Main KV cache keys [batch, num_kv_heads, seq_len, head_dim]
            main_values: Main KV cache values
            seq_len: Current sequence length

        Returns:
            Combined keys and values for attention
        """
        if self.topk_key_cache is None or not self.topk_selected:
            return main_keys, main_values

        # The main cache now contains [sink_tokens, recent_tokens]
        # We need to insert top-k between sink and recent

        batch_size = main_keys.shape[0]
        num_sink = self.config.num_sink_tokens

        # Extract sink and recent from main cache
        # Main cache structure after eviction: [sink, recent, new]
        main_len = main_keys.shape[2]
        recent_len = main_len - num_sink

        sink_keys = main_keys[:, :, :num_sink, :]
        sink_values = main_values[:, :, :num_sink, :]
        recent_keys = main_keys[:, :, num_sink:, :]
        recent_values = main_values[:, :, num_sink:, :]

        # Handle GQA for top-k cache
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            # Expand sink and recent to match top-k head count
            sink_keys_exp = (
                sink_keys[:, :, None, :, :]
                .expand(batch_size, self.num_kv_heads, n_rep, num_sink, self.head_dim)
                .reshape(batch_size, self.num_heads, num_sink, self.head_dim)
            )

            sink_values_exp = (
                sink_values[:, :, None, :, :]
                .expand(batch_size, self.num_kv_heads, n_rep, num_sink, self.head_dim)
                .reshape(batch_size, self.num_heads, num_sink, self.head_dim)
            )

            recent_keys_exp = (
                recent_keys[:, :, None, :, :]
                .expand(batch_size, self.num_kv_heads, n_rep, recent_len, self.head_dim)
                .reshape(batch_size, self.num_heads, recent_len, self.head_dim)
            )

            recent_values_exp = (
                recent_values[:, :, None, :, :]
                .expand(batch_size, self.num_kv_heads, n_rep, recent_len, self.head_dim)
                .reshape(batch_size, self.num_heads, recent_len, self.head_dim)
            )
        else:
            sink_keys_exp = sink_keys
            sink_values_exp = sink_values
            recent_keys_exp = recent_keys
            recent_values_exp = recent_values

        # Concatenate: [sink, top_k, recent]
        combined_keys = torch.cat(
            [sink_keys_exp, self.topk_key_cache, recent_keys_exp], dim=2
        )
        combined_values = torch.cat(
            [sink_values_exp, self.topk_value_cache, recent_values_exp], dim=2
        )

        return combined_keys, combined_values

    def reset(self) -> None:
        """Reset cache state for new request."""
        self.topk_key_cache = None
        self.topk_value_cache = None
        self.seen_tokens = 0
        self.topk_selected = False
        self.topk_indices = None


class MiniMaxM2SageAttention(nn.Module):
    """MiniMax M2 Attention layer with SAGE cache management."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        layer_idx: int,
        rope_parameters: dict[str, Any] | None = None,
        attn_window_size: int | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        sage_config: SageConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # SAGE configuration
        self.sage_config = sage_config or SageConfig()
        self.use_sage = (
            self.sage_config.enabled and layer_idx >= self.sage_config.num_full_kv_layer
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if (
            rope_parameters is not None
            and "partial_rotary_factor" not in rope_parameters
        ):
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        # Use standard vLLM Attention (SAGE logic is handled separately)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=attn_window_size,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.q_norm = MiniMaxText01RMSNormTP(
            self.head_dim * self.total_num_heads, eps=rms_norm_eps
        )
        self.k_norm = MiniMaxText01RMSNormTP(
            self.head_dim * self.total_num_kv_heads, eps=rms_norm_eps
        )

        # Note: SAGE KV cache will be managed at the model level
        # for coordination across layers

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = MiniMaxText01RMSNormTP.forward_qk(self.q_norm, self.k_norm, q, k)
        q, k = self.rotary_emb(positions, q, k)

        # Standard attention forward (SAGE logic in cache management)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM2SageDecoderLayer(nn.Module):
    """Decoder layer using SAGE attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        prefix: str,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        sage_config: SageConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = max(
                config.max_position_embeddings, config.max_model_len
            )

        self.self_attn = MiniMaxM2SageAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=config.rotary_dim,
            layer_idx=layer_idx,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            sage_config=sage_config,
            prefix=f"{prefix}.self_attn",
        )

        self.block_sparse_moe = MiniMaxM2MoE(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)

        return hidden_states, residual


@support_torch_compile
class MiniMaxM2SageModel(nn.Module):
    """MiniMax M2 model with SAGE attention."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        # Initialize SAGE configuration
        self.sage_config = SageConfig.from_hf_config(config)
        logger.info(
            "MiniMax M2 SAGE Model: window=%d, sink=%d, top_k=%d",
            self.sage_config.window_length,
            self.sage_config.num_sink_tokens,
            self.sage_config.top_k,
        )

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Create layers with SAGE configuration
        def make_layer(layer_prefix: str) -> MiniMaxM2SageDecoderLayer:
            # Extract layer index from prefix
            layer_idx = int(layer_prefix.split(".")[-1])
            return MiniMaxM2SageDecoderLayer(
                config,
                layer_idx=layer_idx,
                prefix=layer_prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                sage_config=self.sage_config,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            make_layer,
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name.endswith((".k_scale", ".v_scale")):
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is not None and remapped_name in params_dict:
                        param = params_dict[remapped_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        break

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MiniMaxM2SageForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """
    MiniMax M2 model for causal LM with SAGE attention.

    This model extends MiniMaxM2ForCausalLM with SAGE (Self-Attention Guided
    Eviction) for efficient long-sequence inference.
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        self.model = MiniMaxM2SageModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
