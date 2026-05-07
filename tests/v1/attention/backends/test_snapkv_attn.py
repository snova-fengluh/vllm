# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SnapKV (attention-guided KV cache compression) implementation."""

import pytest
import torch

from vllm.config.snapkv import SnapKVConfig
from vllm.v1.attention.backends.snapkv_attn import (
    _gather_paged_kv,
    _repeat_kv,
    _snapkv_compute_topk,
)
from vllm.v1.attention.backends.snapkv_cache_state import (
    SnapKVRequestState,
    SnapKVRequestStateRegistry,
)
from vllm.v1.kv_cache_interface import SnapKVAttentionSpec


# ---------------------------------------------------------------------------
# SnapKVConfig
# ---------------------------------------------------------------------------


class TestSnapKVConfig:
    """Tests for SnapKVConfig validation."""

    def test_defaults(self):
        cfg = SnapKVConfig()
        assert cfg.enabled is False
        assert cfg.window_length == 8192
        assert cfg.query_window_size == 30
        assert cfg.kernel_size == 13
        assert cfg.pooling == "avgpool"
        assert cfg.num_full_kv_layer == 0

    def test_topk_length(self):
        cfg = SnapKVConfig(window_length=1024, query_window_size=32)
        assert cfg.topk_length == 992

    def test_invalid_window_length(self):
        with pytest.raises(ValueError, match="window_length"):
            SnapKVConfig(enabled=True, window_length=0)

    def test_invalid_query_window_size(self):
        with pytest.raises(ValueError, match="query_window_size"):
            SnapKVConfig(enabled=True, query_window_size=0)

    def test_query_window_too_large(self):
        with pytest.raises(ValueError, match="query_window_size"):
            SnapKVConfig(
                enabled=True, window_length=64, query_window_size=64
            )

    def test_invalid_kernel_size(self):
        with pytest.raises(ValueError, match="kernel_size"):
            SnapKVConfig(enabled=True, kernel_size=4)  # even
        with pytest.raises(ValueError, match="kernel_size"):
            SnapKVConfig(enabled=True, kernel_size=0)

    def test_invalid_pooling(self):
        with pytest.raises(ValueError, match="pooling"):
            SnapKVConfig(enabled=True, pooling="unknown")

    def test_disabled_skips_validation(self):
        # When disabled, even bogus values are accepted.
        cfg = SnapKVConfig(enabled=False, window_length=0, kernel_size=4)
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# SnapKVAttentionSpec
# ---------------------------------------------------------------------------


class TestSnapKVAttentionSpec:
    """Tests for the SnapKVAttentionSpec KV cache spec."""

    def test_basic_spec(self):
        spec = SnapKVAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            window_length=8192,
            query_window_size=32,
        )
        assert spec.window_length == 8192
        assert spec.query_window_size == 32
        assert spec.topk_length == 8192 - 32
        assert spec.kernel_size == 13
        assert spec.pooling == "avgpool"

    def test_memory_bound_uses_window_length(self):
        from unittest.mock import MagicMock

        from vllm.utils.math_utils import cdiv

        spec = SnapKVAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            window_length=8192,
            query_window_size=32,
        )
        vllm_config = MagicMock()
        expected = cdiv(8192, 16) * spec.page_size_bytes
        assert spec.max_memory_usage_bytes(vllm_config) == expected

    def test_merge_returns_snapkv_spec(self):
        specs = [
            SnapKVAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.bfloat16,
                window_length=8192,
                query_window_size=32,
            )
            for _ in range(3)
        ]
        merged = SnapKVAttentionSpec.merge(specs)
        assert isinstance(merged, SnapKVAttentionSpec)
        assert merged.window_length == 8192
        assert merged.query_window_size == 32


# ---------------------------------------------------------------------------
# Per-request state
# ---------------------------------------------------------------------------


class TestSnapKVRequestState:
    """Tests for SnapKVRequestState and SnapKVRequestStateRegistry."""

    def test_request_state_defaults(self):
        state = SnapKVRequestState()
        assert state.compressed_keys_per_layer == {}
        assert state.compressed_values_per_layer == {}
        assert state.prefill_length == 0
        assert state.compressed is False

    def test_request_state_isolation(self):
        registry = SnapKVRequestStateRegistry()
        s1 = registry.get_or_create("req-1")
        s2 = registry.get_or_create("req-2")
        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 4, 8)
        s1.compressed_keys_per_layer[0] = a
        s2.compressed_keys_per_layer[0] = b
        assert torch.equal(
            registry.get_or_create("req-1").compressed_keys_per_layer[0], a
        )
        assert torch.equal(
            registry.get_or_create("req-2").compressed_keys_per_layer[0], b
        )

    def test_registry_prune(self):
        registry = SnapKVRequestStateRegistry()
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
        registry = SnapKVRequestStateRegistry()
        s1 = registry.get_or_create("x")
        s1.compressed = True
        s2 = registry.get_or_create("x")
        assert s2 is s1
        assert s2.compressed is True

    def test_registry_get_returns_none(self):
        registry = SnapKVRequestStateRegistry()
        assert registry.get("missing") is None
        registry.get_or_create("present")
        assert registry.get("present") is not None


# ---------------------------------------------------------------------------
# _repeat_kv
# ---------------------------------------------------------------------------


class TestRepeatKV:
    def test_no_repeat(self):
        seq_len, num_kv, head_dim = 16, 8, 32
        keys = torch.randn(seq_len, num_kv, head_dim)
        values = torch.randn(seq_len, num_kv, head_dim)
        rk, rv = _repeat_kv(keys, values, n_rep=1)
        assert torch.equal(rk, keys)
        assert torch.equal(rv, values)

    def test_gqa_expansion(self):
        seq_len, num_kv, head_dim = 16, 4, 32
        n_rep = 6
        keys = torch.randn(seq_len, num_kv, head_dim)
        values = torch.randn(seq_len, num_kv, head_dim)
        rk, rv = _repeat_kv(keys, values, n_rep=n_rep)
        assert rk.shape == (seq_len, num_kv * n_rep, head_dim)
        assert rv.shape == (seq_len, num_kv * n_rep, head_dim)
        for i in range(num_kv):
            for j in range(n_rep):
                assert torch.equal(rk[:, i * n_rep + j, :], keys[:, i, :])
                assert torch.equal(rv[:, i * n_rep + j, :], values[:, i, :])


# ---------------------------------------------------------------------------
# _snapkv_compute_topk
# ---------------------------------------------------------------------------


class TestSnapKVComputeTopk:
    """Tests for the SnapKV top-k compression algorithm."""

    def test_output_shape(self):
        seq_len, num_q_heads, head_dim = 200, 4, 16
        W = 30
        topk_len = 64
        scale = 1.0 / (head_dim ** 0.5)
        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = torch.randn(seq_len, num_q_heads, head_dim)
        v = torch.randn(seq_len, num_q_heads, head_dim)
        ck, cv = _snapkv_compute_topk(
            q, k, v,
            query_window_size=W, topk_length=topk_len,
            kernel_size=13, pooling="avgpool", scale=scale,
        )
        # Compressed = [topk_past + last_W]
        assert ck.shape == (num_q_heads, topk_len + W, head_dim)
        assert cv.shape == (num_q_heads, topk_len + W, head_dim)

    def test_observation_window_preserved(self):
        """The last W tokens of the prefill must appear verbatim in the
        last W slots of the compressed cache."""
        seq_len, num_q_heads, head_dim = 100, 2, 8
        W = 16
        topk_len = 32
        scale = 1.0
        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = torch.randn(seq_len, num_q_heads, head_dim)
        v = torch.randn(seq_len, num_q_heads, head_dim)
        ck, cv = _snapkv_compute_topk(
            q, k, v,
            query_window_size=W, topk_length=topk_len,
            kernel_size=5, pooling="avgpool", scale=scale,
        )
        # k/v are [seq_len, H, D]; ck/cv are [H, topk+W, D].
        # The last W rows of ck (along dim=1) should equal k[-W:] permuted
        # to [H, W, D].
        expected_k_tail = k[-W:].permute(1, 0, 2)
        expected_v_tail = v[-W:].permute(1, 0, 2)
        assert torch.allclose(ck[:, -W:, :], expected_k_tail)
        assert torch.allclose(cv[:, -W:, :], expected_v_tail)

    def test_topk_picks_high_attention_token(self):
        """If one past position has overwhelmingly high attention, it
        should be present in the compressed cache."""
        seq_len, num_q_heads, head_dim = 60, 1, 4
        W = 10
        topk_len = 5
        scale = 1.0
        # Make all keys uniform except one stand-out.
        q = torch.zeros(seq_len, num_q_heads, head_dim)
        # The W observation queries are aligned with the stand-out key.
        target_idx = 17  # well inside [0, seq_len - W)
        q[-W:, 0, :] = 1.0
        k = torch.zeros(seq_len, num_q_heads, head_dim)
        k[target_idx, 0, :] = 100.0  # huge logit for this position
        v = torch.arange(seq_len, dtype=torch.float32).view(seq_len, 1, 1).expand(
            seq_len, num_q_heads, head_dim
        ).contiguous()

        ck, cv = _snapkv_compute_topk(
            q, k, v,
            query_window_size=W, topk_length=topk_len,
            kernel_size=3, pooling="avgpool", scale=scale,
        )
        # The compressed values from the past portion (first topk_len rows)
        # should include the value-vector that originally came from
        # position `target_idx` (whose value tensor is target_idx).
        past_v = cv[0, :topk_len, 0]  # [topk_len]
        assert (past_v == float(target_idx)).any().item(), (
            f"target idx {target_idx} not selected; got {past_v.tolist()}"
        )

    def test_short_sequence_no_topk(self):
        """When seq_len == W there are no past tokens to select from;
        the compressed cache should just be the observation window."""
        W = 25
        num_q_heads, head_dim = 2, 4
        topk_len = 5
        q = torch.randn(W, num_q_heads, head_dim)
        k = torch.randn(W, num_q_heads, head_dim)
        v = torch.randn(W, num_q_heads, head_dim)
        ck, cv = _snapkv_compute_topk(
            q, k, v,
            query_window_size=W, topk_length=topk_len,
            kernel_size=3, pooling="avgpool", scale=1.0,
        )
        # No past tokens available; output should be [H, W, D]
        assert ck.shape == (num_q_heads, W, head_dim)
        assert cv.shape == (num_q_heads, W, head_dim)

    def test_maxpool(self):
        seq_len, num_q_heads, head_dim = 80, 2, 4
        W = 10
        topk_len = 8
        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = torch.randn(seq_len, num_q_heads, head_dim)
        v = torch.randn(seq_len, num_q_heads, head_dim)
        ck, _ = _snapkv_compute_topk(
            q, k, v,
            query_window_size=W, topk_length=topk_len,
            kernel_size=5, pooling="maxpool", scale=1.0,
        )
        assert ck.shape == (num_q_heads, topk_len + W, head_dim)

    def test_invalid_pooling_raises(self):
        seq_len, num_q_heads, head_dim = 80, 2, 4
        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = torch.randn(seq_len, num_q_heads, head_dim)
        v = torch.randn(seq_len, num_q_heads, head_dim)
        with pytest.raises(ValueError, match="pooling"):
            _snapkv_compute_topk(
                q, k, v,
                query_window_size=10, topk_length=8,
                kernel_size=3, pooling="bogus", scale=1.0,
            )


# ---------------------------------------------------------------------------
# _gather_paged_kv
# ---------------------------------------------------------------------------


class TestGatherPagedKV:
    def test_basic_gather(self):
        num_blocks, block_size = 4, 16
        num_kv_heads, head_size = 2, 8
        kv_cache = torch.randn(
            2, num_blocks, block_size, num_kv_heads, head_size
        )
        block_table_row = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        token_indices = torch.tensor([0, 16, 32, 48], dtype=torch.int64)
        keys, values = _gather_paged_kv(
            kv_cache, block_table_row, token_indices, block_size,
        )
        assert keys.shape == (4, num_kv_heads, head_size)
        assert values.shape == (4, num_kv_heads, head_size)
        assert torch.allclose(keys[0], kv_cache[0, 0, 0])
        assert torch.allclose(values[0], kv_cache[1, 0, 0])
        assert torch.allclose(keys[1], kv_cache[0, 1, 0])
        assert torch.allclose(values[2], kv_cache[1, 2, 0])

    def test_within_block_offsets(self):
        num_blocks, block_size = 2, 16
        num_kv_heads, head_size = 1, 4
        kv_cache = torch.randn(
            2, num_blocks, block_size, num_kv_heads, head_size
        )
        block_table_row = torch.tensor([0, 1], dtype=torch.int32)
        token_indices = torch.tensor([3, 5, 17, 31], dtype=torch.int64)
        keys, _ = _gather_paged_kv(
            kv_cache, block_table_row, token_indices, block_size
        )
        assert torch.allclose(keys[0], kv_cache[0, 0, 3])
        assert torch.allclose(keys[1], kv_cache[0, 0, 5])
        assert torch.allclose(keys[2], kv_cache[0, 1, 1])
        assert torch.allclose(keys[3], kv_cache[0, 1, 15])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
