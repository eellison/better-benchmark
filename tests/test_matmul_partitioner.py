"""matmul_partitioner: the examples of each pattern category."""

from __future__ import annotations

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import partition_node_is_supported
from matmul_partitioner import (
    MatmulLimits,
    classify_pattern_graph,
    get_matmul_regions,
    is_matmul,
)

M = 64
WIDE = MatmulLimits(reduction_limit=256, intermediate_limit=256)


def regions(fn, *inputs, limits=None):
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    return get_matmul_regions(gm, partition_node_is_supported, limits=limits)


def categories(fn, *inputs, limits=None):
    return sorted(r.category for r in regions(fn, *inputs, limits=limits))


def vertical(fn, *inputs, limits=None):
    found = [r for r in regions(fn, *inputs, limits=limits) if r.category == "vertical"]
    assert len(found) <= 1
    return found[0].metrics["vertical"] if found else None


def ln(x):
    var, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
    return (x - mean) * torch.rsqrt(var + 1e-5)


R = torch.randn


# --- 1. prologue -> MM -------------------------------------------------------


def test_gather_prologue():
    idx = torch.randint(0, 100, (M,))
    assert categories(lambda t, i, w: t[i] @ w, R(100, 32), idx, R(32, 48)) == ["prologue"]


def test_branching_pointwise_prologue():
    def fn(mask, a, x, c, w):
        return torch.where(mask, a * x + 1, torch.sigmoid(c + 5)) @ w

    args = (torch.rand(M, 32) > 0.5, R(M, 32), R(M, 32), R(M, 32), R(32, 48))
    assert categories(fn, *args) == ["prologue"]


# --- 2. MM -> epilogue -------------------------------------------------------


def test_bias_relu_epilogue():
    assert categories(lambda x, w, b: torch.relu(x @ w + b), R(M, 32), R(32, 48), R(48)) == [
        "epilogue"
    ]


def test_epilogue_with_side_inputs_and_gather():
    def fn(x, w, b, i):
        return torch.relu(x @ w) + b[i]

    (region,) = regions(fn, R(M, 32), R(32, 48), R(100, 48), torch.randint(0, 100, (M,)))
    assert region.category == "epilogue"
    assert any("index" in str(n.target) for n in region)


def test_prologue_and_epilogue_is_its_own_category():
    def fn(x, g, b, w, res):
        return torch.relu(ln(x) * g + b) @ w + res

    args = (R(M, 32), R(32), R(32), R(32, 48), R(M, 48))
    # Only the combined category: the MM is not also "prologue" or "epilogue".
    # (norm -> MM is additionally a vertical chain.)
    assert categories(fn, *args) == ["prologue_epilogue", "vertical"]


def test_bare_mm_is_not_a_region():
    assert categories(lambda x, w: x @ w, R(M, 32), R(32, 48)) == []


# --- 3. horizontal -----------------------------------------------------------


def test_qkv_horizontal_group():
    found = regions(lambda x, a, b, c: (x @ a, x @ b, x @ c), R(M, 32), *[R(32, 48)] * 3)
    assert [r.category for r in found] == ["horizontal"]
    assert len(found[0].outputs) == 3


def test_horizontal_group_stops_at_three():
    def fn(x, a, b, c, d):
        return x @ a, x @ b, x @ c, x @ d

    (group,) = regions(fn, R(M, 32), *[R(32, 48)] * 4)
    assert sum(is_matmul(n) for n in group) == 3


def test_horizontal_needs_equal_output_shapes_but_not_equal_k():
    def fn(x, y, a, b):
        return x @ a, x[:, :16] @ b

    # x[:, :16] is a slice, not a transparent op: different shared input.
    assert categories(fn, R(M, 32), R(M, 16), R(32, 48), R(16, 48)) == []
    fn2 = lambda x, a, b: (x @ a, x @ b)
    assert categories(fn2, R(M, 32), R(32, 48), R(32, 24)) == []


def test_horizontal_join_is_horizontal_combine():
    def fn(x, a, b):
        return torch.relu(x @ a) + torch.sigmoid(x @ b)

    # The group is only "horizontal_combine"; each MM is also an epilogue.
    assert categories(fn, R(M, 32), R(32, 48), R(32, 48)) == [
        "epilogue",
        "epilogue",
        "horizontal_combine",
    ]


def test_horizontal_with_separate_epilogues_is_not_combine():
    def fn(x, a, b):
        return torch.relu(x @ a), torch.sigmoid(x @ b)

    found = [r for r in regions(fn, R(M, 32), R(32, 48), R(32, 48)) if is_mm_group(r)]
    assert [r.category for r in found] == ["horizontal"]
    assert len(found[0].outputs) == 2


def test_partial_combine_of_three_is_horizontal_combine():
    def fn(x, a, b, c):
        return torch.nn.functional.silu(x @ a) * (x @ b), x @ c

    found = [r for r in regions(fn, R(M, 32), *[R(32, 48)] * 3) if is_mm_group(r)]
    assert [r.category for r in found] == ["horizontal_combine"]
    assert sum(is_matmul(n) for n in found[0]) == 3


def is_mm_group(region):
    return sum(is_matmul(n) for n in region) >= 2


# --- 4. vertical -------------------------------------------------------------


def test_gather_ln_mm_residual_ln():
    def chain(t, i, w, res, g, b):
        return ln((ln(t[i]) * g + b) @ w + res)

    args = (R(1000, 128), torch.randint(0, 1000, (M,)), R(128, 256), R(M, 256), R(128), R(128))
    # With 256 limits: var_mean -> MM -> var_mean, slice = row.
    summary = vertical(chain, *args, limits=WIDE)
    assert [s["op"] for s in summary["stages"]] == ["var_mean", "mm", "var_mean"]
    assert summary["slice_axes"] == [M]
    assert [s["F"] for s in summary["stages"]] == [1, 256, 0]
    # Default 128: the second norm reduces 256 and stays outside.
    summary = vertical(chain, *args)
    assert [s["op"] for s in summary["stages"]] == ["var_mean", "mm"]


def test_mm_relu_matvec_needs_f_256():
    fn = lambda a, b, v: torch.relu(a @ b) @ v
    args = (R(M, 128), R(128, 256), R(256))
    assert vertical(fn, *args) is None
    summary = vertical(fn, *args, limits=WIDE)
    assert summary["stages"] == [{"op": "mm", "R": 128, "F": 256}, {"op": "mv", "R": 256, "F": 0}]


def test_grouped_sum_slices_by_row_and_group():
    summary = vertical(lambda a, b: (a @ b).view(M, 128, 32).sum(-1), R(M, 128), R(128, 4096))
    assert sorted(summary["slice_axes"]) == [M, 128]
    assert summary["stages"] == [{"op": "mm", "R": 128, "F": 32}, {"op": "sum", "R": 32, "F": 0}]


def test_mm_chain_slices_by_row():
    fn = lambda a, b, c, d: (a @ b) @ c @ d
    summary = vertical(fn, R(M, 128), R(128, 64), R(64, 128), R(128, 64))
    assert summary["slice_axes"] == [M]
    assert [s["F"] for s in summary["stages"]] == [64, 128, 0]


def test_first_mm_reduction_size_is_checked():
    fn = lambda a, b, c: (a @ b) @ c
    assert vertical(fn, R(M, 512), R(512, 128), R(128, 64)) is None


def test_chunked_gate_needs_whole_intermediate():
    def fn(a, b):
        u, v = (a @ b).chunk(2, dim=-1)
        return (u.sigmoid() * v).sum(-1)

    assert vertical(fn, R(M, 128), R(128, 512)) is None


def test_softmax_axis_is_not_independent():
    def fn(a, b):
        s = a @ b
        e = (s - s.amax(-1, keepdim=True)).exp()
        return e / e.sum(-1, keepdim=True)

    summary = vertical(fn, R(M, 64), R(64, 128))
    assert summary["slice_axes"] == [M]
    assert [s["F"] for s in summary["stages"]] == [128, 1, 0]


# --- classification of standalone graphs ----------------------------------


def test_classify_pattern_graph_matches_extraction():
    cases = [
        (lambda t, i, w: t[i] @ w, (R(100, 32), torch.randint(0, 100, (M,)), R(32, 48))),
        (lambda x, w, b: torch.relu(x @ w + b), (R(M, 32), R(32, 48), R(48))),
        (lambda x, a, b: (x @ a, x @ b), (R(M, 32), R(32, 48), R(32, 48))),
        (
            lambda x, a, b: (x @ a).relu() * (x @ b),
            (R(M, 32), R(32, 48), R(32, 48)),
        ),
        (lambda a, b: (a @ b).view(M, 128, 32).sum(-1), (R(M, 128), R(128, 4096))),
    ]
    for fn, args in cases:
        # The region covering the whole function (MMs may also be epilogues).
        region = max(regions(fn, *args), key=len)
        gm = make_fx(fn, tracing_mode="fake")(*args)
        assert classify_pattern_graph(gm) == region.category


def test_swiglu_join_reached_from_either_branch():
    def fn(x, wg, wu):
        g = (x @ wg).view(2, 32, 48)
        u = (x @ wu).view(2, 32, 48)
        return torch.nn.functional.silu(g.float()).to(torch.bfloat16) * u

    args = (R(M, 32).bfloat16(), R(32, 48).bfloat16(), R(32, 48).bfloat16())
    found = [r for r in regions(fn, *args) if is_mm_group(r)]
    assert [r.category for r in found] == ["horizontal_combine"]
    assert len(found[0].outputs) == 1


def test_causal_mask_does_not_alias_score_axes():
    def fn(q, k, v):
        s = q @ k.T
        i = torch.arange(M)
        s = s.masked_fill(i[:, None] < i[None, :], -1e4)
        e = (s - s.amax(-1, keepdim=True)).exp()
        return (e / e.sum(-1, keepdim=True)) @ v

    summary = vertical(fn, R(M, 32), R(M, 32), R(M, 32))
    assert summary["slice_axes"] == [M]
    assert [s["op"] for s in summary["stages"]] == ["mm", "amax", "sum", "mm"]


def test_side_input_from_graph_inputs_is_absorbed():
    def fn(x, w, saved):
        gelu_grad = 0.5 * (1 + torch.erf(saved * 0.7071)) + saved * torch.exp(-saved * saved)
        return (x @ w) * gelu_grad

    (region,) = regions(fn, R(M, 32), R(32, 48), R(M, 48))
    assert region.category == "epilogue"
    assert any("erf" in str(n.target) for n in region)


def test_side_input_from_intermediates_stays_outside():
    def fn(x, w, h, g):
        normed = ln(h) * g  # computed from a reduction: stays an input
        return x @ w + normed

    found = [r for r in regions(fn, R(M, 32), R(32, 48), R(M, 48), R(48)) if r.category != "vertical"]
    assert [r.category for r in found] == ["epilogue"]
    assert not any("rsqrt" in str(n.target) for n in found[0])


