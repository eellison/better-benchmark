"""Shared normalization for benchmark result identity and provenance."""

from __future__ import annotations

_KNOWN_BENCHMARK_CONFIG_DEFAULTS = {
    "combo_kernels": False,
    "combo_kernel_per_subkernel_blocks": False,
    "benchmark_combo_kernel": False,
    "combo_kernels_pointwise_only": False,
    "multi_kernel": 0,
    "persistent_reduction_mode": "default",
    # bench_parallel enables CD unless --no-cd is passed. Missing metadata in
    # legacy result files therefore means the historical default, True.
    "coordinate_descent": True,
    "strict_gpu_lock": False,
}


def semantic_benchmark_config(config: dict | None) -> dict:
    """Drop execution placement while retaining result-affecting settings."""
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError("benchmark configuration must be an object")

    normalized = {
        key: config.get(key)
        for key, default in _KNOWN_BENCHMARK_CONFIG_DEFAULTS.items()
        if config.get(key, default) != default
    }
    for key in ("inductor_config", "extra_inductor_config"):
        value = config.get(key)
        if value:
            if not isinstance(value, dict):
                raise ValueError(f"benchmark config {key} must be an object")
            normalized["inductor_config"] = dict(value)
    # Result-affecting: a sweep with a backend registered is not the same run
    # as one without, so results from the two must not merge into each other.
    worker_init = config.get("worker_init")
    if worker_init:
        if not isinstance(worker_init, list):
            raise ValueError("benchmark config worker_init must be an array")
        normalized["worker_init"] = list(worker_init)
    return normalized
