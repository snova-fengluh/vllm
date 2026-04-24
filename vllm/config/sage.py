# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SAGE (Self-Attention Guided Eviction) configuration for bounded KV cache."""

from vllm.config.utils import config


@config
class SageConfig:
    """Configuration for SAGE attention mechanism.

    SAGE enables efficient long-sequence inference by maintaining a bounded
    KV cache through attention-guided token selection. It combines StreamLLM-style
    windowing with top-k selection of historically important tokens.

    The KV cache structure is: [sink_tokens] + [top_k_tokens] + [recent_tokens]
    """

    enabled: bool = False
    """Whether to enable SAGE attention for bounded KV cache management."""

    window_length: int = 8192
    """Total KV cache window size (sink + top_k + recent tokens).
    This determines the maximum number of tokens stored in the KV cache."""

    num_sink_tokens: int = 4
    """Number of initial tokens to always preserve (sink tokens).
    These tokens are kept regardless of attention scores."""

    top_k: int = 512
    """Number of historically important tokens to select via attention scores.
    After prefill, the top-k tokens with highest attention weights
    (excluding sink and recent tokens) are preserved."""

    num_full_kv_layer: int = 0
    """Number of early layers to use full KV cache (no SAGE eviction).
    Set to 0 to apply SAGE to all layers."""

    def __post_init__(self):
        if self.enabled:
            # Validate configuration
            recent_window = (
                self.window_length - self.num_sink_tokens - self.top_k
            )
            if recent_window <= 0:
                raise ValueError(
                    f"Invalid SAGE config: window_length ({self.window_length}) "
                    f"must be > num_sink_tokens ({self.num_sink_tokens}) + "
                    f"top_k ({self.top_k}). Got recent_window = {recent_window}"
                )
            if self.num_sink_tokens < 0:
                raise ValueError(
                    f"num_sink_tokens must be >= 0, got {self.num_sink_tokens}"
                )
            if self.top_k < 0:
                raise ValueError(f"top_k must be >= 0, got {self.top_k}")
            if self.num_full_kv_layer < 0:
                raise ValueError(
                    f"num_full_kv_layer must be >= 0, got {self.num_full_kv_layer}"
                )
