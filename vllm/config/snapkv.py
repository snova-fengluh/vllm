# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SnapKV configuration for attention-guided KV cache compression."""

from vllm.config.utils import config


@config
class SnapKVConfig:
    """Configuration for SnapKV attention mechanism.

    SnapKV compresses the KV cache after prefilling by using the attention
    pattern of the last ``query_window_size`` queries (the "observation
    window") to vote on which past tokens are most important. The selected
    top-k tokens together with the observation window form the compressed
    cache of size ``window_length``.

    The compressed KV cache structure is:
        [top_k_important_tokens] + [observation_window]
    """

    enabled: bool = False
    """Whether to enable SnapKV attention for compressed KV cache."""

    window_length: int = 8192
    """Total compressed KV cache size after prefill (top_k + observation
    window). This is the budget kept from the prefill stage; new tokens
    generated during decode are appended on top."""

    query_window_size: int = 30
    """Number of recent prefill queries used as observers when voting on
    which past tokens are important."""

    kernel_size: int = 13
    """Kernel size of the 1D pooling applied to the importance scores
    before top-k selection. Must be odd."""

    pooling: str = "avgpool"
    """Pooling strategy applied to importance scores. Either ``avgpool`` or
    ``maxpool``."""

    num_full_kv_layer: int = 0
    """Number of early layers that keep the full (uncompressed) KV cache.
    Layers with ``layer_idx < num_full_kv_layer`` skip SnapKV entirely.
    Set to 0 to apply SnapKV to all layers."""

    def __post_init__(self):
        if self.enabled:
            if self.window_length <= 0:
                raise ValueError(
                    f"window_length must be > 0, got {self.window_length}"
                )
            if self.query_window_size <= 0:
                raise ValueError(
                    f"query_window_size must be > 0, "
                    f"got {self.query_window_size}"
                )
            if self.query_window_size >= self.window_length:
                raise ValueError(
                    f"query_window_size ({self.query_window_size}) must be "
                    f"< window_length ({self.window_length})"
                )
            if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
                raise ValueError(
                    f"kernel_size must be a positive odd integer, "
                    f"got {self.kernel_size}"
                )
            if self.pooling not in ("avgpool", "maxpool"):
                raise ValueError(
                    f"pooling must be 'avgpool' or 'maxpool', "
                    f"got {self.pooling!r}"
                )
            if self.num_full_kv_layer < 0:
                raise ValueError(
                    f"num_full_kv_layer must be >= 0, "
                    f"got {self.num_full_kv_layer}"
                )

    @property
    def topk_length(self) -> int:
        """Number of top-k past tokens kept (= window_length - query_window_size)."""
        return self.window_length - self.query_window_size

    def compute_hash(self) -> str:
        """Provide a hash that uniquely identifies all the SnapKV configs
        that affect the structure of the computation graph."""
        from vllm.config.utils import get_hash_factors, hash_factors

        ignored_factors: set[str] = set()
        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)
