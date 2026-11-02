"""Reject singleton FX reduction axes, independently of slice R/F metrics."""

from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import partition_node_is_supported
from matmul_partitioner import (
    get_matmul_regions,
    is_reduction,
    singleton_reduction,
    tensor_shape,
)


def vertical(fn, *args):
    gm = make_fx(fn, tracing_mode="fake")(*args)
    return get_matmul_regions(gm, partition_node_is_supported, categories=("vertical",))


@pytest.mark.parametrize(
    "reduce",
    [
        lambda x: x.sum([0, 1]),
        lambda x: x.mean([0, 1]),
        lambda x: torch.var_mean(x, [0, 1], correction=0),
        lambda x: x.sum([-2, -1]),
        lambda x: x.sum(),
        lambda x: x.sum([]),
        lambda x: x.amax(dim=[0, 1], keepdim=True),
    ],
)
def test_mixed_reduced_axes_include_a_singleton_even_when_total_r_is_six(reduce):
    gm = make_fx(reduce, tracing_mode="fake")(torch.randn(1, 6))
    shapes = {n: tensor_shape(n) for n in gm.graph.nodes}
    stage = next(n for n in gm.graph.nodes if is_reduction(n))
    assert singleton_reduction(stage, shapes)


def test_unreduced_singleton_axis_is_not_rejected():
    gm = make_fx(lambda x: x.sum(dim=1), tracing_mode="fake")(torch.randn(1, 6))
    shapes = {n: tensor_shape(n) for n in gm.graph.nodes}
    stage = next(n for n in gm.graph.nodes if is_reduction(n))
    assert not singleton_reduction(stage, shapes)


@pytest.mark.parametrize("first_k,hidden", [(1, 6), (6, 1)])
def test_either_mm_k_one_rejects_the_vertical_chain(first_k, hidden):
    assert (
        vertical(
            lambda x, a, b: (x @ a) @ b,
            torch.randn(4, first_k),
            torch.randn(first_k, hidden),
            torch.randn(hidden, 8),
        )
        == []
    )


def test_mm_n_one_without_reducing_that_axis_remains_eligible():
    found = vertical(lambda x, w: (x @ w).sum(0), torch.randn(4, 8), torch.randn(8, 1))
    assert found
    assert found[0].metrics["vertical"]["stages"] == [
        {"op": "mm", "R": 8, "F": 4},
        {"op": "sum", "R": 4, "F": 0},
    ]


def test_reducing_mm_n_one_is_rejected():
    assert (
        vertical(lambda x, w: (x @ w).mean(-1), torch.randn(4, 8), torch.randn(8, 1))
        == []
    )


def test_f_one_is_not_a_rejection_rule():
    def fn(x, w):
        var, mean = torch.var_mean(x, -1, keepdim=True, correction=0)
        return ((x - mean) * torch.rsqrt(var + 1e-5)) @ w

    found = vertical(fn, torch.randn(4, 8), torch.randn(8, 6))
    assert found and found[0].metrics["vertical"]["stages"][0]["F"] == 1


def test_clean_prefix_can_survive_a_singleton_reduction():
    found = vertical(
        lambda x, w: (x @ w).sum(-1, keepdim=True).mean(-1),
        torch.randn(4, 8),
        torch.randn(8, 6),
    )
    assert found
    assert [s["op"] for s in found[0].metrics["vertical"]["stages"]] == ["mm", "sum"]


def test_clean_suffix_can_survive_a_singleton_reduction():
    found = vertical(
        lambda x, w: (x.sum(1) @ w).sum(-1), torch.randn(4, 1, 8), torch.randn(8, 6)
    )
    assert found
    assert [s["op"] for s in found[0].metrics["vertical"]["stages"]] == ["mm", "sum"]


@pytest.mark.parametrize("combine", [False, True])
def test_k_one_does_not_change_horizontal_policy(combine):
    def fn(x, a, b):
        first, second = x @ a, x @ b
        return first + second if combine else (first, second)

    gm = make_fx(fn, tracing_mode="fake")(
        torch.randn(4, 1), torch.randn(1, 6), torch.randn(1, 6)
    )
    found = get_matmul_regions(
        gm, partition_node_is_supported, categories=("horizontal", "horizontal_combine")
    )
    assert [r.category for r in found] == [
        "horizontal_combine" if combine else "horizontal"
    ]
