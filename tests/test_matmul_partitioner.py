"""matmul_partitioner: the examples of each pattern category."""

from __future__ import annotations

import pytest
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


# --- normal BB partition absorption ----------------------------------------


def normal_regions(fn, *inputs):
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    return get_matmul_regions(gm, partition_node_is_supported, categories=("prologue", "epilogue"))


def test_pointwise_partition_absorbs_consumer_mm_first():
    (region,) = normal_regions(lambda x, w: torch.cos(torch.sin(x) @ w), R(M, 32), R(32, 48))
    assert region.category == "prologue"
    assert any(n.target is torch.ops.aten.sin.default for n in region)
    assert not any(n.target is torch.ops.aten.cos.default for n in region)
    assert sum(is_matmul(n) for n in region) == 1


def test_addmm_bias_only_partition_does_not_absorb_consumer():
    assert normal_regions(
        lambda bias, x, w: torch.addmm(torch.tanh(bias), x, w),
        R(16), R(128, 16), R(16, 16),
    ) == []


def test_addmm_bias_dtype_cast_is_not_a_prologue():
    assert normal_regions(
        lambda bias, x, w: torch.addmm(
            torch.ops.prims.convert_element_type.default(bias, torch.bfloat16), x, w
        ),
        R(16), R(128, 16).bfloat16(), R(16, 16).bfloat16(),
    ) == []


def test_addmm_bias_only_leaves_mm_available_for_an_epilogue():
    (region,) = normal_regions(
        lambda bias, x, w: torch.cos(torch.addmm(torch.tanh(bias), x, w)),
        R(16), R(128, 16), R(16, 16),
    )
    assert region.category == "epilogue"
    assert not any(n.target is torch.ops.aten.tanh.default for n in region)


def test_addmm_bias_partition_cannot_claim_mm_before_matrix_input_partition():
    def fn(bias, x, w):
        b = torch.tanh(bias)
        h = torch.sin(x)
        return torch.addmm(b, h, w)

    (region,) = normal_regions(fn, R(16), R(128, 16), R(16, 16))
    assert region.category == "prologue"
    assert any(n.target is torch.ops.aten.sin.default for n in region)
    assert not any(n.target is torch.ops.aten.tanh.default for n in region)


@pytest.mark.parametrize("operand", [0, 1])
def test_addmm_pointwise_on_either_multiplied_operand_remains_prologue(operand):
    def fn(bias, x, w):
        h = torch.tanh(x) if operand == 0 else x
        weight = torch.tanh(w) if operand == 1 else w
        return (h if operand == 0 else weight), torch.addmm(bias, h, weight)

    (region,) = normal_regions(fn, R(16), R(128, 16), R(16, 16))
    assert region.category == "prologue"
    mm = next(n for n in region if is_matmul(n))
    transformed = next(n for n in region if n.target is torch.ops.aten.tanh.default)
    assert mm.args[1 + operand] is transformed
    assert len(region.outputs) == 2


@pytest.mark.parametrize("packet", [torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm])
def test_multiplication_operand_binding_excludes_bias_and_supports_keywords(packet):
    from mm_partition_absorption import _matmul_inputs

    graph = torch.fx.Graph()
    bias, left, right = [graph.placeholder(name) for name in ("bias", "left", "right")]
    kwargs = {"mat1": left, "mat2": right, "self": bias} if packet is torch.ops.aten.addmm else {
        "self": left, "mat2": right,
    }
    node = graph.call_function(packet.default, kwargs=kwargs)
    assert _matmul_inputs(node) == [left, right]


def test_reduction_pointwise_partition_is_not_split_for_a_prologue():
    assert normal_regions(lambda x, w: ln(x) @ w, R(M, 32), R(32, 48)) == []


def test_gather_partition_is_not_a_pointwise_prologue():
    idx = torch.randint(0, 100, (M,))
    assert normal_regions(lambda t, i, w: t[i] @ w, R(100, 32), idx, R(32, 48)) == []


def test_epilogue_keeps_the_whole_reduction_partition():
    def fn(x, w):
        z = x @ w
        e = (z - z.amax(-1, keepdim=True)).exp()
        return (e / e.sum(-1, keepdim=True)).sum(-1)

    (region,) = normal_regions(fn, R(M, 32), R(32, 48))
    assert region.category == "epilogue"
    assert sum(torch.Tag.reduction in getattr(n.target, "tags", ()) for n in region) == 3


def test_absorbed_partitions_and_mms_are_not_reused():
    def fn(x, a, b):
        h = torch.sin(x)
        first, second = h @ a, h @ b
        return torch.cos(first) + second, h

    found = normal_regions(fn, R(M, 32), R(32, 48), R(32, 48))
    assert sorted(r.category for r in found) == ["epilogue", "prologue"]
    assert not set(found[0]) & set(found[1])
    assert all(sum(is_matmul(n) for n in r) == 1 for r in found)
    assert any(len(r.outputs) == 2 for r in found)  # h also escapes to the other MM/output


def test_single_mm_category_filter_does_not_change_absorption_order():
    gm = make_fx(lambda x, w: torch.cos(torch.sin(x) @ w), tracing_mode="fake")(R(M, 32), R(32, 48))
    assert get_matmul_regions(gm, partition_node_is_supported, categories=("epilogue",)) == []


def test_bare_mm_is_not_a_region():
    assert categories(lambda x, w: x @ w, R(M, 32), R(32, 48)) == []


# --- horizontal ------------------------------------------------------------


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

    # Horizontal windows stay independent of the disjoint absorption pass.
    found = regions(fn, R(M, 32), R(32, 48), R(32, 48))
    assert [r.category for r in found if is_mm_group(r)] == ["horizontal_combine"]
    assert sum(r.category == "epilogue" for r in found) == 1


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
        (lambda x, w: torch.sin(x) @ w, (R(M, 32), R(32, 48))),
        (lambda x, w, b: torch.relu(x @ w + b), (R(M, 32), R(32, 48), R(48))),
        (lambda x, a, b: (x @ a, x @ b), (R(M, 32), R(32, 48), R(32, 48))),
        (
            lambda x, a, b: (x @ a).relu() * (x @ b),
            (R(M, 32), R(32, 48), R(32, 48)),
        ),
        (lambda a, b: (a @ b).view(M, 128, 32).sum(-1), (R(M, 128), R(128, 4096))),
    ]
    from capture_hook import extract_partition_subgraph

    for fn, args in cases:
        gm = make_fx(fn, tracing_mode="fake")(*args)
        for region in get_matmul_regions(gm, partition_node_is_supported):
            subset, _, _ = extract_partition_subgraph(region, gm)
            expected = region.category if region.category in {"prologue", "epilogue"} else None
            assert classify_pattern_graph(subset, expected=expected) == region.category


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


def test_epilogue_keeps_reduction_side_work_in_its_original_partition():
    def fn(x, w, h, g):
        return x @ w + ln(h) * g

    found = normal_regions(fn, R(M, 32), R(32, 48), R(M, 48), R(48))
    assert [r.category for r in found] == ["epilogue"]
    assert any("rsqrt" in str(n.target) for n in found[0])


