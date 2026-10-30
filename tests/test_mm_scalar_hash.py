"""MM pattern hashes include scalar literals; other pattern hashes do not."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import (
    compute_dag_signature,
    compute_partition_pattern,
    get_fusion_partitions,
    pattern_hash_for_subgraph,
)


def trace(fn, *shapes):
    return make_fx(fn, tracing_mode="fake")(*[torch.randn(s) for s in shapes])


def mm_pattern(gm):
    # The largest region holds the literal (e.g. the var_mean projection is
    # only inside the vertical region; the prologue region takes it as input).
    region = max(get_fusion_partitions(gm, only_include_mm=True), key=len)
    return compute_partition_pattern(region, gm)


def scale_by_stat(index):
    def fn(x, w):
        stats = torch.var_mean(x, dim=1, keepdim=True, correction=0)
        return ((x * stats[index]) @ w).relu()

    return fn


# Pairs of MM regions that differ only in one scalar literal.
SCALAR_PAIRS = {
    "int_divisor": (
        lambda x, w: ((x / 49) @ w).relu(),
        lambda x, w: ((x / 64) @ w).relu(),
        [(4, 8), (8, 8)],
    ),
    "float_divisor": (
        lambda x, w: ((x / 49.0) @ w).relu(),
        lambda x, w: ((x / 64.0) @ w).relu(),
        [(4, 8), (8, 8)],
    ),
    "epsilon": (
        lambda v, x, w: ((x * torch.rsqrt(v + 1e-3)) @ w).relu(),
        lambda v, x, w: ((x * torch.rsqrt(v + 1e-5)) @ w).relu(),
        [(1, 8), (4, 8), (8, 8)],
    ),
    "keyword_argument": (
        lambda b, x, w: torch.addmm(b, x, w, beta=2.0).relu(),
        lambda b, x, w: torch.addmm(b, x, w, beta=3.0).relu(),
        [(8,), (4, 8), (8, 8)],
    ),
    "tuple_index": (scale_by_stat(0), scale_by_stat(1), [(4, 8), (8, 8)]),
}


@pytest.mark.parametrize("name", SCALAR_PAIRS)
def test_mm_pattern_hash_changes_with_a_scalar_literal(name):
    first_fn, second_fn, shapes = SCALAR_PAIRS[name]
    first = mm_pattern(trace(first_fn, *shapes))
    second = mm_pattern(trace(second_fn, *shapes))
    assert first["pattern_hash"] != second["pattern_hash"]
    assert first["shape_hash"] == second["shape_hash"]


def test_non_mm_pattern_hash_still_ignores_scalars():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1.0))
    graph.output(graph.call_function(torch.ops.aten.mul.Tensor, (add, 3.0)))
    gm = torch.fx.GraphModule({}, graph)
    assert compute_dag_signature(gm) == [
        ("input", 0),
        ("aten.add.Tensor", [("node", 0, 0)]),
        ("aten.mul.Tensor", [("node", 1, 0)]),
        ("output", 2),
    ]
    for value in [3.0, 5.0]:
        traced = trace(lambda x: (x + 1.0) * value, (4, 8))
        (region,) = get_fusion_partitions(traced)
        pattern = compute_partition_pattern(region, traced)
        assert pattern["pattern_hash"] == pattern_hash_for_subgraph(gm)


def test_mm_pattern_hash_survives_retracing():
    pattern = mm_pattern(trace(lambda x, w: ((x / 49.0) @ w).relu(), (4, 8), (8, 8)))
    retraced = trace(pattern["sub_gm"], (4, 8), (8, 8))
    assert pattern_hash_for_subgraph(retraced, scalar_sensitive=True) == pattern["pattern_hash"]


def test_mm_pattern_hash_ignores_unused_tuple_projections():
    # Retracing max_pool2d emits a dead getitem for the unused indices.
    gm = make_fx(
        lambda x, w: (torch.nn.functional.max_pool2d(x, 2).flatten(1) @ w).relu(),
        tracing_mode="fake",
    )(torch.randn(2, 3, 8, 8), torch.randn(48, 5))
    trimmed = copy.deepcopy(gm)
    trimmed.graph.eliminate_dead_code()
    trimmed.recompile()
    assert pattern_hash_for_subgraph(gm, scalar_sensitive=True) == pattern_hash_for_subgraph(
        trimmed, scalar_sensitive=True
    )


def test_nested_tuple_outputs_hash_by_structure_not_order():
    def graph(outputs):
        g = torch.fx.Graph()
        x = g.placeholder("x")
        y = g.call_function(torch.ops.aten.relu.default, (x,))
        g.output(outputs(x, y))
        return torch.fx.GraphModule({}, g)

    first = graph(lambda x, y: (y, (x, y)))
    reordered = graph(lambda x, y: ((x, y), y))
    different = graph(lambda x, y: (x, (y, y)))
    assert pattern_hash_for_subgraph(first) == pattern_hash_for_subgraph(reordered)
    assert pattern_hash_for_subgraph(first) != pattern_hash_for_subgraph(different)
