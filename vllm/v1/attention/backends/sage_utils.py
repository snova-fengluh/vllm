# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for SAGE (Self-Attention Guided Eviction) attention."""

import torch
from torch import Tensor


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """
    Expand KV heads for GQA (Grouped Query Attention).

    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_kv_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim).

    Args:
        hidden_states: Input tensor of shape [batch, num_kv_heads, seqlen, head_dim]
        n_rep: Number of repetitions (num_query_heads // num_kv_heads)

    Returns:
        Expanded tensor of shape [batch, num_kv_heads * n_rep, seqlen, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def sage_topk_selection(
    query: Tensor,
    keys: Tensor,
    top_k: int,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """
    Perform top-k selection for SAGE attention.

    Selects the top-k most important tokens based on attention scores
    computed as Q @ K^T.

    Args:
        query: Query tensor of shape [batch, num_heads, 1, head_dim]
        keys: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        top_k: Number of top tokens to select
        scale: Attention scaling factor (typically 1/sqrt(head_dim))

    Returns:
        Tuple of (topk_indices, topk_scores):
            - topk_indices: Shape [batch, num_heads, top_k]
            - topk_scores: Shape [batch, num_heads, top_k]
    """
    # Compute attention scores: [batch, num_heads, 1, seq_len]
    # Note: For GQA, keys may have fewer heads than queries.
    # The caller should expand keys if needed before calling this function.
    scores = torch.matmul(query, keys.transpose(-2, -1)) * scale

    # Remove the query dimension: [batch, num_heads, seq_len]
    scores = scores.squeeze(-2)

    # Select top-k indices: [batch, num_heads, top_k]
    topk_scores, topk_indices = torch.topk(
        scores, k=top_k, dim=-1, largest=True, sorted=False
    )

    return topk_indices, topk_scores


def gather_topk_kv(
    keys: Tensor,
    values: Tensor,
    topk_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Gather top-k KV pairs based on selected indices.

    Args:
        keys: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        values: Value tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        topk_indices: Indices of shape [batch, num_heads, top_k]

    Returns:
        Tuple of (topk_keys, topk_values):
            - topk_keys: Shape [batch, num_heads, top_k, head_dim]
            - topk_values: Shape [batch, num_heads, top_k, head_dim]
    """
    batch, num_heads, top_k = topk_indices.shape
    head_dim = keys.shape[-1]
    num_kv_heads = keys.shape[1]

    # Expand indices for gathering: [batch, num_heads, top_k, head_dim]
    gather_indices = topk_indices.unsqueeze(-1).expand(
        batch, num_heads, top_k, head_dim
    )

    # For GQA, we need to handle the case where num_heads != num_kv_heads
    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        # Expand keys and values to match query head count
        keys = repeat_kv(keys, n_rep)
        values = repeat_kv(values, n_rep)

    # Gather top-k keys and values
    topk_keys = torch.gather(keys, dim=2, index=gather_indices)
    topk_values = torch.gather(values, dim=2, index=gather_indices)

    return topk_keys, topk_values
