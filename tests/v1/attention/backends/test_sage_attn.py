# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SAGE (Self-Attention Guided Eviction) attention implementation."""

import pytest
import torch

from vllm.v1.attention.backends.sage_cache_state import (
    SageRequestState,
    SageRequestStateRegistry,
)
from vllm.v1.attention.backends.sage_utils import (
    gather_topk_kv,
    repeat_kv,
    sage_topk_selection,
)
from vllm.v1.kv_cache_interface import SageAttentionSpec


# ---------------------------------------------------------------------------
# sage_utils helpers (unchanged)
# ---------------------------------------------------------------------------


class TestRepeatKV:
    """Tests for the repeat_kv utility function."""

    def test_no_repeat(self):
        batch, num_kv_heads, seq_len, head_dim = 2, 8, 16, 64
        hidden_states = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        result = repeat_kv(hidden_states, n_rep=1)
        assert result.shape == hidden_states.shape
        assert torch.equal(result, hidden_states)

    def test_repeat_gqa(self):
        batch, num_kv_heads, seq_len, head_dim = 2, 8, 16, 64
        n_rep = 6
        hidden_states = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        result = repeat_kv(hidden_states, n_rep=n_rep)
        assert result.shape == (batch, num_kv_heads * n_rep, seq_len, head_dim)
        for i in range(num_kv_heads):
            for j in range(n_rep):
                assert torch.equal(
                    result[:, i * n_rep + j, :, :],
                    hidden_states[:, i, :, :],
                )

    def test_repeat_mqa(self):
        batch, num_kv_heads, seq_len, head_dim = 2, 1, 16, 64
        n_rep = 32
        hidden_states = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        result = repeat_kv(hidden_states, n_rep=n_rep)
        assert result.shape == (batch, n_rep, seq_len, head_dim)


class TestSageTopkSelection:
    """Tests for the sage_topk_selection function."""

    def test_basic_selection(self):
        batch, num_heads, seq_len, head_dim = 1, 4, 100, 64
        top_k = 10
        scale = 1.0 / (head_dim ** 0.5)
        query = torch.randn(batch, num_heads, 1, head_dim)
        keys = torch.randn(batch, num_heads, seq_len, head_dim)
        topk_indices, topk_scores = sage_topk_selection(
            query, keys, top_k, scale)
        assert topk_indices.shape == (batch, num_heads, top_k)
        assert topk_scores.shape == (batch, num_heads, top_k)
        assert (topk_indices >= 0).all()
        assert (topk_indices < seq_len).all()

    def test_selection_correctness(self):
        batch, num_heads, seq_len, head_dim = 1, 1, 10, 4
        top_k = 3
        scale = 1.0
        query = torch.ones(batch, num_heads, 1, head_dim)
        keys = torch.zeros(batch, num_heads, seq_len, head_dim)
        keys[0, 0, 2, :] = 10.0
        keys[0, 0, 5, :] = 8.0
        keys[0, 0, 7, :] = 6.0
        topk_indices, _ = sage_topk_selection(query, keys, top_k, scale)
        selected = set(topk_indices[0, 0].tolist())
        assert selected == {2, 5, 7}


class TestGatherTopkKV:
    """Tests for the gather_topk_kv function."""

    def test_basic_gather(self):
        batch, num_heads, seq_len, head_dim = 1, 4, 100, 64
        top_k = 10
        keys = torch.randn(batch, num_heads, seq_len, head_dim)
        values = torch.randn(batch, num_heads, seq_len, head_dim)
        topk_indices = torch.randint(0, seq_len, (batch, num_heads, top_k))
        topk_keys, topk_values = gather_topk_kv(keys, values, topk_indices)
        assert topk_keys.shape == (batch, num_heads, top_k, head_dim)
        assert topk_values.shape == (batch, num_heads, top_k, head_dim)

    def test_gather_correctness(self):
        batch, num_heads, seq_len, head_dim = 1, 2, 10, 4
        keys = (
            torch.arange(seq_len)
            .float()
            .view(1, 1, seq_len, 1)
            .expand(batch, num_heads, seq_len, head_dim)
        )
        values = keys.clone()
        topk_indices = torch.tensor([[[1, 5, 9], [2, 4, 6]]])
        topk_keys, _ = gather_topk_kv(keys, values, topk_indices)
        assert topk_keys[0, 0, 0, 0].item() == 1.0
        assert topk_keys[0, 0, 1, 0].item() == 5.0
        assert topk_keys[0, 0, 2, 0].item() == 9.0
        assert topk_keys[0, 1, 0, 0].item() == 2.0
        assert topk_keys[0, 1, 1, 0].item() == 4.0
        assert topk_keys[0, 1, 2, 0].item() == 6.0


# ---------------------------------------------------------------------------
# SageAttentionSpec
# ---------------------------------------------------------------------------


class TestSageAttentionSpec:
    """Tests for the SageAttentionSpec KV cache spec."""

    def test_basic_spec(self):
        spec = SageAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            window_length=8192,
            num_sink_tokens=4,
            top_k=512,
        )
        assert spec.window_length == 8192
        assert spec.num_sink_tokens == 4
        assert spec.top_k == 512
        assert spec.recent_window_size == 8192 - 4 - 512

    def test_sage_attention_spec_memory_bound(self):
        """max_memory_usage_bytes should be based on window_length only."""
        from unittest.mock import MagicMock

        spec = SageAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            window_length=8192,
            num_sink_tokens=4,
            top_k=512,
        )
        # Build a mock vllm_config – the method should NOT reference
        # max_model_len (unlike FullAttentionSpec).
        vllm_config = MagicMock()
        # Compute expected: cdiv(8192, 16) * page_size_bytes
        from vllm.utils.math_utils import cdiv

        expected = cdiv(8192, 16) * spec.page_size_bytes
        assert spec.max_memory_usage_bytes(vllm_config) == expected

    def test_sage_attention_spec_merge(self):
        """Merging SageAttentionSpec should return SageAttentionSpec."""
        specs = [
            SageAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.bfloat16,
                window_length=8192,
                num_sink_tokens=4,
                top_k=512,
            )
            for _ in range(5)
        ]
        merged = SageAttentionSpec.merge(specs)
        assert isinstance(merged, SageAttentionSpec)
        assert merged.window_length == 8192
        assert merged.num_sink_tokens == 4
        assert merged.top_k == 512


# ---------------------------------------------------------------------------
# Per-request state
# ---------------------------------------------------------------------------


class TestSageRequestState:
    """Tests for SageRequestState and SageRequestStateRegistry."""

    def test_request_state_defaults(self):
        state = SageRequestState()
        assert state.topk_indices_per_layer == {}
        assert state.prefill_done is False

    def test_request_state_isolation(self):
        """Two requests must accumulate independent topk_indices."""
        registry = SageRequestStateRegistry()
        s1 = registry.get_or_create("req-1")
        s2 = registry.get_or_create("req-2")

        idx1 = torch.tensor([[0, 1, 2], [3, 4, 5]])
        idx2 = torch.tensor([[10, 11, 12], [13, 14, 15]])
        s1.topk_indices_per_layer[0] = idx1
        s2.topk_indices_per_layer[0] = idx2

        assert torch.equal(
            registry.get_or_create("req-1").topk_indices_per_layer[0], idx1
        )
        assert torch.equal(
            registry.get_or_create("req-2").topk_indices_per_layer[0], idx2
        )

    def test_registry_prune(self):
        registry = SageRequestStateRegistry()
        registry.get_or_create("a")
        registry.get_or_create("b")
        registry.get_or_create("c")
        assert len(registry) == 3

        registry.prune({"a", "c"})
        assert len(registry) == 2
        assert "a" in registry
        assert "b" not in registry
        assert "c" in registry

    def test_registry_get_or_create_idempotent(self):
        registry = SageRequestStateRegistry()
        s1 = registry.get_or_create("x")
        s1.prefill_done = True
        s2 = registry.get_or_create("x")
        assert s2 is s1
        assert s2.prefill_done is True


# ---------------------------------------------------------------------------
# _topk_select_for_row
# ---------------------------------------------------------------------------


class TestTopkSelectForRow:
    """Tests for _topk_select_for_row helper."""

    def test_shape_mha(self):
        """MHA: num_heads == num_kv_heads."""
        from vllm.v1.attention.backends.sage_attn import _topk_select_for_row

        num_kv_heads = 8
        head_dim = 64
        top_k = 10
        num_candidates = 100
        scale = 1.0 / (head_dim ** 0.5)

        query_row = torch.randn(num_kv_heads, head_dim)  # MHA: q_per_kv=1
        cand_keys = torch.randn(num_candidates, num_kv_heads, head_dim)

        indices = _topk_select_for_row(
            query_row, cand_keys, top_k, scale, num_queries_per_kv=1
        )
        assert indices.shape == (num_kv_heads, top_k)
        assert (indices >= 0).all()
        assert (indices < num_candidates).all()

    def test_shape_gqa(self):
        """GQA: num_heads = 4 * num_kv_heads."""
        from vllm.v1.attention.backends.sage_attn import _topk_select_for_row

        num_kv_heads = 2
        num_q_per_kv = 4
        num_heads = num_kv_heads * num_q_per_kv
        head_dim = 32
        top_k = 5
        num_candidates = 50
        scale = 1.0 / (head_dim ** 0.5)

        query_row = torch.randn(num_heads, head_dim)
        cand_keys = torch.randn(num_candidates, num_kv_heads, head_dim)

        indices = _topk_select_for_row(
            query_row, cand_keys, top_k, scale,
            num_queries_per_kv=num_q_per_kv)
        assert indices.shape == (num_kv_heads, top_k)

    def test_correctness(self):
        """The top-1 selected index should be the highest-scoring candidate."""
        from vllm.v1.attention.backends.sage_attn import _topk_select_for_row

        num_kv_heads = 1
        head_dim = 4
        scale = 1.0

        query_row = torch.ones(1, head_dim)  # MHA, 1 head
        cand_keys = torch.zeros(10, num_kv_heads, head_dim)
        cand_keys[7, 0, :] = 100.0  # Make candidate 7 the clear winner

        indices = _topk_select_for_row(
            query_row, cand_keys, top_k=1, scale=scale,
            num_queries_per_kv=1)
        assert indices[0, 0].item() == 7


# ---------------------------------------------------------------------------
# _gather_paged_kv
# ---------------------------------------------------------------------------


class TestGatherPagedKV:
    """Tests for _gather_paged_kv helper."""

    def test_basic_gather(self):
        from vllm.v1.attention.backends.sage_attn import _gather_paged_kv

        num_blocks = 4
        block_size = 16
        num_kv_heads = 2
        head_size = 8

        kv_cache = torch.randn(2, num_blocks, block_size, num_kv_heads,
                                head_size)
        # Block table: tokens 0..63 map to blocks 0,1,2,3
        block_table_row = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        token_indices = torch.tensor([0, 16, 32, 48], dtype=torch.int64)

        keys, values = _gather_paged_kv(
            kv_cache, block_table_row, token_indices, block_size)

        assert keys.shape == (4, num_kv_heads, head_size)
        assert values.shape == (4, num_kv_heads, head_size)

        # Verify correctness for first token (block 0, offset 0)
        assert torch.allclose(keys[0], kv_cache[0, 0, 0])
        assert torch.allclose(values[0], kv_cache[1, 0, 0])


# ---------------------------------------------------------------------------
# growing_recent_window flag wiring
# ---------------------------------------------------------------------------


class TestGrowingRecentWindowFlag:
    """The sage_growing_recent_window flag must propagate from CLI/EngineArgs
    through SageConfig to the attention impl, and the per-request anchor
    field must exist on SageRequestState."""

    def test_sage_config_default_is_false(self):
        from vllm.config.sage import SageConfig

        cfg = SageConfig()
        assert cfg.growing_recent_window is False

    def test_sage_config_round_trip(self):
        from vllm.config.sage import SageConfig

        cfg = SageConfig(
            enabled=True,
            window_length=8192,
            num_sink_tokens=4,
            top_k=512,
            growing_recent_window=True,
        )
        assert cfg.growing_recent_window is True

    def test_engine_args_cli_propagates(self):
        """`--sage-growing-recent-window` must surface through EngineArgs
        and end up on the constructed SageConfig."""
        from vllm.engine.arg_utils import EngineArgs

        args = EngineArgs(
            model="facebook/opt-125m",
            sage_enabled=True,
            sage_growing_recent_window=True,
        )
        assert args.sage_growing_recent_window is True

    def test_request_state_anchor_field_defaults_none(self):
        """SageRequestState.recent_window_start must default to None so the
        gather path can detect the not-yet-frozen case."""
        state = SageRequestState()
        assert state.recent_window_start is None

    def test_request_state_anchor_is_per_instance(self):
        """The anchor is per-request (different prompt lengths produce
        different anchors)."""
        registry = SageRequestStateRegistry()
        s1 = registry.get_or_create("req-a")
        s2 = registry.get_or_create("req-b")
        s1.recent_window_start = 2325
        s2.recent_window_start = 1325
        assert registry.get_or_create("req-a").recent_window_start == 2325
        assert registry.get_or_create("req-b").recent_window_start == 1325


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
