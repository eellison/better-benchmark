"""MM repro identities retain baked scalar semantics without migrating PW IDs."""
from __future__ import annotations

import json
import operator

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import (
    _CaptureState,
    compute_dag_signature,
    compute_partition_pattern,
    get_fusion_partitions,
    pattern_hash_for_subgraph,
)
from matmul_partitioner import MatmulRegion


def trace(fn, *shapes):
    return make_fx(fn, tracing_mode="fake")(*[torch.randn(s) for s in shapes])


def mm_pattern(gm):
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert isinstance(regions[0], MatmulRegion)
    return compute_partition_pattern(regions[0], gm)


@pytest.mark.parametrize("divisor", [49, 49.0])
def test_mm_division_scalar_changes_identity(divisor):
    first = mm_pattern(trace(lambda x, w: ((x / divisor) @ w).relu(), (4, 8), (8, 8)))
    second = mm_pattern(trace(lambda x, w: ((x / 64) @ w).relu(), (4, 8), (8, 8)))
    assert first["pattern_hash"] != second["pattern_hash"]
    assert first["legacy_pattern_hash"] == second["legacy_pattern_hash"]
    assert first["shape_hash"] == second["shape_hash"]
    assert first["pattern_hash_mode"] == "scalar_literals_v1"


def test_mm_normalization_epsilon_changes_identity():
    def pattern(eps):
        return mm_pattern(trace(
            lambda variance, x, w: ((x * torch.rsqrt(variance + eps)) @ w).relu(),
            (1, 8), (4, 8), (8, 8),
        ))

    source, other = pattern(0.001), pattern(1e-5)
    assert source["pattern_hash"] != other["pattern_hash"]
    assert source["legacy_pattern_hash"] == other["legacy_pattern_hash"]
    assert source["shape_hash"] == other["shape_hash"]
    assert source["legacy_pattern_hash"] == pattern_hash_for_subgraph(source["sub_gm"])
    assert source["pattern_hash"] == pattern_hash_for_subgraph(
        source["sub_gm"], scalar_sensitive=True,
    )


def test_mm_scalar_keyword_changes_identity():
    def pattern(beta):
        return mm_pattern(trace(
            lambda bias, x, w: torch.addmm(bias, x, w, beta=beta).relu(),
            (8,), (4, 8), (8, 8),
        ))

    first, second = pattern(2.0), pattern(3.0)
    assert first["pattern_hash"] != second["pattern_hash"]
    assert first["legacy_pattern_hash"] == second["legacy_pattern_hash"]


def test_mm_getitem_index_changes_identity():
    def pattern(index):
        def fn(x, w):
            stats = torch.var_mean(x, dim=1, keepdim=True, correction=0)
            return ((x * stats[index]) @ w).relu()

        return mm_pattern(trace(fn, (4, 8), (8, 8)))

    variance, mean = pattern(0), pattern(1)
    assert variance["pattern_hash"] != mean["pattern_hash"]
    for pat, index in [(variance, 0), (mean, 1)]:
        projections = [n for n in pat["sub_gm"].graph.nodes
                       if n.target is operator.getitem and n.users]
        assert len(projections) == 1
        assert projections[0].args[1] == index

    # make_fx emits both tuple projections, including an unused one. Check
    # a single-projection DAG too so the regression cannot pass merely
    # because the incidental ordering of that dead projection forks a hash.
    def graph(index):
        g = torch.fx.Graph()
        x, w = g.placeholder("x"), g.placeholder("w")
        stats = g.call_function(torch.ops.aten.var_mean.correction, (x, [1]),
                                {"correction": 0, "keepdim": True})
        value = g.call_function(operator.getitem, (stats, index))
        scaled = g.call_function(torch.ops.aten.mul.Tensor, (x, value))
        mm = g.call_function(torch.ops.aten.mm.default, (scaled, w))
        g.output(mm)
        return torch.fx.GraphModule({}, g)

    first, second = graph(0), graph(1)
    assert pattern_hash_for_subgraph(first) == pattern_hash_for_subgraph(second)
    assert pattern_hash_for_subgraph(first, scalar_sensitive=True) != pattern_hash_for_subgraph(
        second, scalar_sensitive=True,
    )


def test_non_mm_default_scalar_identity_and_signature_remain_legacy():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1.0))
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (add, 3.0))
    graph.output(mul)
    gm = torch.fx.GraphModule({}, graph)
    assert compute_dag_signature(gm) == [
        ("input", 0),
        ("aten.add.Tensor", [("node", 0, 0)]),
        ("aten.mul.Tensor", [("node", 1, 0)]),
        ("output", 2),
    ]
    changed = trace(lambda x: (x + 1.0) * 5.0, (4, 8))
    assert pattern_hash_for_subgraph(gm) == pattern_hash_for_subgraph(changed)
    def bind_scalar(value):
        return lambda x: (x + 1.0) * value

    for value in [3.0, 5.0]:
        traced = trace(bind_scalar(value), (4, 8))
        regions = get_fusion_partitions(traced)
        assert len(regions) == 1
        pattern = compute_partition_pattern(regions[0], traced)
        assert pattern["pattern_hash"] == pattern_hash_for_subgraph(gm)
        assert "legacy_pattern_hash" not in pattern
        assert "pattern_hash_mode" not in pattern


def test_scalar_hash_is_stable_after_mm_retrace():
    gm = trace(lambda x, w: ((x / 49.0) @ w).relu(), (4, 8), (8, 8))
    pattern = mm_pattern(gm)
    sub = pattern["sub_gm"]
    retraced = trace(sub, (4, 8), (8, 8))
    assert pattern_hash_for_subgraph(retraced, scalar_sensitive=True) == pattern["pattern_hash"]


def test_mm_capture_serializes_scalar_hash_mode_and_legacy_id(tmp_path):
    gm = trace(lambda x, w: ((x / 49) @ w).relu(), (4, 8), (8, 8))
    expected = mm_pattern(gm)
    state = _CaptureState(str(tmp_path), validate=False, only_include_mm=True)
    observations = state.process_graph(gm)
    state.finalize()
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["n_captured"] == 1
    assert index["n_dropped"] == 0
    for entry in [observations[0], index["captured"][0]]:
        assert entry["pattern_hash_mode"] == expected["pattern_hash_mode"]
        assert entry["pattern_hash"] == expected["pattern_hash"]
        assert entry["legacy_pattern_hash"] == expected["legacy_pattern_hash"]
