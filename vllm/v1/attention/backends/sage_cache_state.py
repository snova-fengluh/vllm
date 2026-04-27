# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request SAGE state for tracking top-k indices across layers."""

from dataclasses import dataclass, field

import torch


@dataclass
class SageRequestState:
    """Per-request SAGE state.

    Each active request has its own state tracking which tokens were
    selected as top-k important for each layer, and whether the
    prefill→decode transition has occurred.
    """

    # Per-layer top-k indices.  Each tensor is [num_kv_heads, top_k] int64
    # holding token-position indices relative to the request's KV sequence.
    topk_indices_per_layer: dict[int, torch.Tensor] = field(default_factory=dict)

    # True once the first decode step after prefill has been processed.
    prefill_done: bool = False


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
