"""Size limits applied to the actual standalone MM repro graph."""

from __future__ import annotations

import torch
from mm_corpus_io import admits, filter_patterns, graph_metrics
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import get_fusion_partitions
from matmul_partitioner import MatmulLimits

# The limits curate_mm_corpus applies to exported repros (the defaults).
FINAL_LIMITS = MatmulLimits()


def test_filter_keeps_every_record_within_the_limits_without_a_count_cap():
    at_limit = {
        "mm": 3,
        "compute": 32,
        "nodes": 48,
        "inputs": 12,
        "outputs": 4,
        "contraction_depth": 8,
    }
    assert admits(at_limit, FINAL_LIMITS)
    for key, value in at_limit.items():
        assert not admits({**at_limit, key: value + 1}, FINAL_LIMITS)
    records = [{"id": i, "metrics": at_limit} for i in range(350)]
    records.append({"id": "big", "metrics": {**at_limit, "mm": 4}})
    assert [r["id"] for r in filter_patterns(records, FINAL_LIMITS)] == list(range(350))


def test_graph_metrics_follow_value_dependencies():
    gm = make_fx(
        lambda x, y, a, b, c: (((x @ a).relu() + y @ b) @ c).relu(), tracing_mode="fake"
    )(*(torch.randn(8, 64) for _ in range(2)), *(torch.randn(64, 64) for _ in range(3)))
    metrics = graph_metrics(gm)
    assert metrics["mm"] == 3
    assert metrics["contraction_depth"] == 2
    assert metrics["inputs"] == 5 and metrics["outputs"] == 1


def test_gather_norm_mm_residual_norm_fits_the_final_limits():
    def norm(x, gamma, beta):
        xf = x.float()
        mean = xf.sum(-1, keepdim=True) / x.size(-1)
        centered = xf - mean
        variance = (centered * centered).sum(-1, keepdim=True) / x.size(-1)
        normalized = centered * torch.rsqrt(variance + 1e-5)
        return (normalized * gamma.float() + beta.float()).to(x.dtype)

    def fn(table, ids, w, residual, pg, pb, qg, qb):
        return norm(norm(table.index_select(0, ids), pg, pb) @ w + residual, qg, qb)

    bf16 = dict(dtype=torch.bfloat16)
    gm = make_fx(fn, tracing_mode="fake")(
        torch.randn(64, 128, **bf16),
        torch.arange(8),
        torch.randn(128, 128, **bf16),
        torch.randn(8, 128, **bf16),
        *(torch.randn(128, **bf16) for _ in range(4)),
    )
    assert admits(graph_metrics(gm), FINAL_LIMITS)
    assert any(
        r.metrics["contraction_depth"] == 5 and admits(r.metrics, FINAL_LIMITS)
        for r in get_fusion_partitions(gm, only_include_mm=True)
    )
