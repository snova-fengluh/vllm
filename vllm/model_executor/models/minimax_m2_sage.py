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
from vllm.config import CacheConfig, ModelConfig, SageConfig, VllmConfig
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
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    SageAttentionSpec,
    get_kv_quant_mode,
)

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
class SageModelConfig:
    """Local configuration for SAGE attention in the model.

    This class is used internally by the model. It can be created from:
    1. CLI arguments via vllm.config.SageConfig (preferred)
    2. HuggingFace config JSON (fallback for backward compatibility)
    """

    enabled: bool = True
    window_length: int = SAGE_DEFAULT_WINDOW_LENGTH
    num_sink_tokens: int = SAGE_DEFAULT_NUM_SINK_TOKENS
    top_k: int = SAGE_DEFAULT_TOP_K
    num_full_kv_layer: int = 0  # Layers that use full KV cache (no SAGE)

    @classmethod
    def from_cli_config(cls, sage_config: SageConfig) -> "SageModelConfig":
        """Create from CLI config (vllm.config.SageConfig)."""
        return cls(
            enabled=sage_config.enabled,
            window_length=sage_config.window_length,
            num_sink_tokens=sage_config.num_sink_tokens,
            top_k=sage_config.top_k,
            num_full_kv_layer=sage_config.num_full_kv_layer,
        )

    @classmethod
    def from_hf_config(cls, config: PretrainedConfig) -> "SageModelConfig":
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


class SageAttention(Attention):
    """Attention subclass that returns SageAttentionSpec for SAGE layers.

    For layers whose ``layer_idx < sage_config.num_full_kv_layer``, the
    standard ``FullAttentionSpec`` is returned (full KV cache, no SAGE).
    Otherwise a ``SageAttentionSpec`` is returned so that the KV cache
    budget is bounded by ``window_length``.
    """

    def __init__(
        self,
        *args: Any,
        sage_config: SageModelConfig,
        layer_idx: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sage_config = sage_config
        self.layer_idx = layer_idx

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Layers below the full-KV threshold use the standard spec.
        if self.layer_idx < self.sage_config.num_full_kv_layer:
            return super().get_kv_cache_spec(vllm_config)

        block_size = vllm_config.cache_config.block_size
        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
        return SageAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=quant_mode,
            window_length=self.sage_config.window_length,
            num_sink_tokens=self.sage_config.num_sink_tokens,
            top_k=self.sage_config.top_k,
            num_full_kv_layer=self.sage_config.num_full_kv_layer,
        )


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
        sage_config: SageModelConfig | None = None,
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
        self.sage_config = sage_config or SageModelConfig()

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

        # SageAttention overrides get_kv_cache_spec to return
        # SageAttentionSpec for SAGE layers (bounded by window_length).
        self.attn = SageAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=attn_window_size,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            sage_config=self.sage_config,
            layer_idx=layer_idx,
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
        sage_config: SageModelConfig | None = None,
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
        # Prefer CLI config (vllm_config.sage_config) over HF config
        if vllm_config.sage_config is not None:
            self.sage_config = SageModelConfig.from_cli_config(vllm_config.sage_config)
            logger.info("Using SAGE config from CLI arguments")
        else:
            self.sage_config = SageModelConfig.from_hf_config(config)
            logger.info("Using SAGE config from model config.json")

        logger.info(
            "SAGE MODEL INITIALIZED: window=%d  sink=%d  top_k=%d  "
            "full_kv_layers=%d  recent=%d",
            self.sage_config.window_length,
            self.sage_config.num_sink_tokens,
            self.sage_config.top_k,
            self.sage_config.num_full_kv_layer,
            self.sage_config.recent_window_size,
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
        def make_layer(prefix: str) -> MiniMaxM2SageDecoderLayer:
            # Extract layer index from prefix
            layer_idx = int(prefix.split(".")[-1])
            return MiniMaxM2SageDecoderLayer(
                config,
                layer_idx=layer_idx,
                prefix=prefix,
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

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
