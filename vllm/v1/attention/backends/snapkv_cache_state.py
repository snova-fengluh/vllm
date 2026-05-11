# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request SnapKV state for tracking compressed KV cache across layers.

Following the sambaLLMs SnapKV reference implementation, the compressed K/V
tensors are stored at the **query head level** (not KV head level). With GQA,
different query heads within the same group may pick different observation
positions as important; storing per-query-head allows top-k selection to be
independent across heads.

Memory cost per request per layer:
    2 * num_query_heads * window_length * head_dim * dtype_size
    e.g., for 32 query heads, window_length=8192, head_dim=128, bf16:
    2 * 32 * 8192 * 128 * 2 = 128 MB per layer
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SnapKVRequestState:
    """Per-request SnapKV state.

    Each active request has its own state tracking the compressed K/V cache
    (top-k past + observation window) for each layer. Tensors are stored at
    query-head granularity to support GQA models.
    """

    # Per-layer compressed keys: {layer_idx: [num_query_heads, window_length, head_dim]}
    # Contains [top_k_past_keys] + [observation_window_keys].
    compressed_keys_per_layer: dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-layer compressed values: same layout as keys.
    compressed_values_per_layer: dict[int, torch.Tensor] = field(default_factory=dict)

    # Sequence length at the moment the request transitioned from prefill to
    # decode. Used to identify positions of decode tokens written into the
    # paged KV cache after compression.
    prefill_length: int = 0

    # True once compression has been performed for this request.
    compressed: bool = False

    # --- Chunked-prefill support ---

    # True while the request is still going through prefill chunks.
    in_prefill: bool = False

    # Total KV length accumulated so far during prefill (updated each chunk).
    prefill_seq_len: int = 0

    # Per-layer saved observation queries (the last W queries from the most
    # recent prefill chunk, accumulated across chunks so that the final W
    # queries of the full prefill are available for compression).
    # {layer_idx: [<=W, num_query_heads, head_dim]}
    observation_queries_per_layer: dict[int, torch.Tensor] = field(
        default_factory=dict
    )


class SnapKVRequestStateRegistry:
    """Registry of per-request SnapKV states, held by the metadata builder.

    Keyed by ``request_id: str``.
    """

    def __init__(self) -> None:
        self._states: dict[str, SnapKVRequestState] = {}

    def get_or_create(self, request_id: str) -> SnapKVRequestState:
        """Return the state for *request_id*, creating one if needed."""
        state = self._states.get(request_id)
        if state is None:
            state = SnapKVRequestState()
            self._states[request_id] = state
        return state

    def get(self, request_id: str) -> SnapKVRequestState | None:
        """Return the state for *request_id* if present, else ``None``."""
        return self._states.get(request_id)

    def prune(self, active_request_ids: set[str]) -> None:
        """Drop states for requests that are no longer active."""
        stale = self._states.keys() - active_request_ids
        for rid in stale:
            del self._states[rid]

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._states
