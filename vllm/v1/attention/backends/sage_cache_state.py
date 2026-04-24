# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SAGE cache state manager for tracking per-layer top-k cache."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class SageLayerCacheState:
    """Per-layer SAGE cache state."""

    # Top-k KV cache for this layer
    # Shape: [batch, num_heads, top_k, head_dim]
    topk_key_cache: Tensor | None = None
    topk_value_cache: Tensor | None = None

    # Whether top-k selection has been performed for this layer
    topk_selected: bool = False

    # Indices of the top-k tokens (for debugging/analysis)
    topk_indices: Tensor | None = None


@dataclass
class SageCacheState:
    """
    Manages per-layer SAGE cache state across requests.

    This class tracks the top-k KV cache for each layer, which is separate
    from the main paged KV cache used by vLLM. The top-k cache stores
    historically important tokens selected via attention-based scoring.
    """

    num_layers: int
    window_length: int
    num_sink_tokens: int
    top_k: int

    # Per-layer cache states
    layer_states: dict[int, SageLayerCacheState] = field(default_factory=dict)

    # Total number of tokens seen so far (across all requests)
    seen_tokens: int = 0

    # Whether SAGE eviction has been triggered (sequence exceeded window)
    eviction_triggered: bool = False

    def __post_init__(self):
        # Initialize layer states
        for layer_idx in range(self.num_layers):
            self.layer_states[layer_idx] = SageLayerCacheState()

    def get_topk_cache(self, layer_idx: int) -> tuple[Tensor | None, Tensor | None]:
        """
        Get the top-k KV cache for a specific layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Tuple of (topk_key_cache, topk_value_cache), or (None, None) if
            top-k selection has not been performed yet.
        """
        if layer_idx not in self.layer_states:
            return None, None

        state = self.layer_states[layer_idx]
        return state.topk_key_cache, state.topk_value_cache

    def set_topk_cache(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
        indices: Tensor | None = None,
    ) -> None:
        """
        Set the top-k KV cache for a specific layer.

        Args:
            layer_idx: Index of the layer
            keys: Top-k keys of shape [batch, num_heads, top_k, head_dim]
            values: Top-k values of shape [batch, num_heads, top_k, head_dim]
            indices: Optional indices of the selected tokens
        """
        if layer_idx not in self.layer_states:
            self.layer_states[layer_idx] = SageLayerCacheState()

        state = self.layer_states[layer_idx]
        state.topk_key_cache = keys
        state.topk_value_cache = values
        state.topk_selected = True
        state.topk_indices = indices

    def is_topk_selected(self, layer_idx: int) -> bool:
        """Check if top-k selection has been performed for a layer."""
        if layer_idx not in self.layer_states:
            return False
        return self.layer_states[layer_idx].topk_selected

    def should_trigger_eviction(self, current_seq_len: int) -> bool:
        """
        Check if SAGE eviction should be triggered.

        Eviction is triggered when the sequence length exceeds the window length.

        Args:
            current_seq_len: Current sequence length

        Returns:
            True if eviction should be triggered
        """
        return current_seq_len > self.window_length

    def reset_for_new_request(self) -> None:
        """Reset cache state for a new request."""
        self.seen_tokens = 0
        self.eviction_triggered = False
        for layer_idx in self.layer_states:
            self.layer_states[layer_idx] = SageLayerCacheState()

    def get_recent_window_size(self) -> int:
        """
        Get the size of the recent token window.

        The recent window is: window_length - num_sink_tokens - top_k
        """
        return self.window_length - self.num_sink_tokens - self.top_k

    @property
    def total_cache_size(self) -> int:
        """
        Get the total number of tokens that can be stored in the SAGE cache.

        This is: num_sink_tokens + top_k + recent_window_size = window_length
        """
        return self.window_length


def create_sage_cache_state(
    num_layers: int,
    window_length: int = 8192,
    num_sink_tokens: int = 4,
    top_k: int = 512,
) -> SageCacheState:
    """
    Create a new SAGE cache state manager.

    Args:
        num_layers: Number of layers in the model
        window_length: Total cache window size (sink + top_k + recent)
        num_sink_tokens: Number of initial tokens to always preserve
        top_k: Number of historically important tokens to select

    Returns:
        Initialized SageCacheState
    """
    # Validate configuration
    assert num_sink_tokens >= 0, "num_sink_tokens must be non-negative"
    assert top_k > 0, "top_k must be positive"
    assert window_length > num_sink_tokens + top_k, (
        f"window_length ({window_length}) must be greater than "
        f"num_sink_tokens ({num_sink_tokens}) + top_k ({top_k})"
    )

    return SageCacheState(
        num_layers=num_layers,
        window_length=window_length,
        num_sink_tokens=num_sink_tokens,
        top_k=top_k,
    )
