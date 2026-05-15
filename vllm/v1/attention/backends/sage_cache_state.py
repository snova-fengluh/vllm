# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request SAGE state for tracking top-k KV cache across layers.

Following the sambaLLMs SAGE implementation, we store the actual top-k K/V
tensors at the **query head level** (not KV head level). This is necessary
because with GQA, different query heads within the same group may select
different tokens as important, and FlashAttention requires the same sequence
length for all heads.

Memory cost per request per layer:
    2 * num_query_heads * top_k * head_dim * dtype_size
    e.g., for 32 query heads, top_k=512, head_dim=128, bf16:
    2 * 32 * 512 * 128 * 2 = 8 MB per layer
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SageRequestState:
    """Per-request SAGE state.

    Each active request has its own state tracking the top-k K/V cache
    for each layer. The K/V tensors are stored at query head level to
    support GQA models where different query heads may select different
    important tokens.
    """

    # Per-layer top-k keys: {layer_idx: [num_query_heads, top_k, head_dim]}
    topk_keys_per_layer: dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-layer top-k values: {layer_idx: [num_query_heads, top_k, head_dim]}
    topk_values_per_layer: dict[int, torch.Tensor] = field(default_factory=dict)

    # True once the first decode step after prefill has been processed.
    prefill_done: bool = False

    # Absolute token position (within this request's sequence) of the first
    # token in the recent window, captured at the first long-decode step.
    # Used only when SageConfig.growing_recent_window is True. When set, the
    # recent window for this request is [recent_window_start, seq_len - 1]
    # and grows by one token per decode step. None when the streaming
    # (fixed-length) policy is in use, or before the first long decode has
    # happened.
    recent_window_start: int | None = None


class SageRequestStateRegistry:
    """Registry of per-request SAGE states, held by the metadata builder.

    Keyed by ``request_id: str``.
    """

    def __init__(self) -> None:
        self._states: dict[str, SageRequestState] = {}

    def get_or_create(self, request_id: str) -> SageRequestState:
        """Return the state for *request_id*, creating one if needed."""
        state = self._states.get(request_id)
        if state is None:
            state = SageRequestState()
            self._states[request_id] = state
        return state

    def prune(self, active_request_ids: set[str]) -> None:
        """Drop states for requests that are no longer active."""
        stale = self._states.keys() - active_request_ids
        for rid in stale:
            del self._states[rid]

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._states
