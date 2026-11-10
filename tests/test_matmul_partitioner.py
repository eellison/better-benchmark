"""matmul_partitioner: the examples of each pattern category."""

from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import get_fusion_partitions, partition_node_is_supported
from matmul_partitioner import (
    classify_pattern_graph,
    get_matmul_regions,
    is_matmul,
    is_reduction_or_scan,
    op_name,
)

M = 64


def regions(fn, *inputs, limits=None):
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    return get_matmul_regions(gm, partition_node_is_supported, limits=limits)


def categories(fn, *inputs, limits=None):
    return sorted(r.category for r in regions(fn, *inputs, limits=limits))


def ln(x):
    var, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
    return (x - mean) * torch.rsqrt(var + 1e-5)


R = torch.randn


# --- normal BB partition absorption ----------------------------------------


def normal_regions(fn, *inputs):
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    return get_matmul_regions(gm, partition_node_is_supported, categories=("prologue", "epilogue"))


def test_partition_absorbs_producer_mm_before_consumer_mm():
    (region,) = normal_regions(lambda x, w: torch.cos(torch.sin(x) @ w), R(M, 32), R(32, 48))
    assert region.category == "epilogue"
    assert any(n.target is torch.ops.aten.cos.default for n in region)
    assert not any(n.target is torch.ops.aten.sin.default for n in region)
    assert sum(is_matmul(n) for n in region) == 1


def test_pointwise_partition_between_mms_is_the_producer_epilogue():
    # relu(x @ w1) @ w2: prologue-first gave only prologue [relu, mm_1].
    (region,) = normal_regions(
        lambda x, w1, w2: torch.relu(x @ w1) @ w2, R(M, 32), R(32, 48), R(48, 16),
    )
    assert region.category == "epilogue"
    (mm,) = [n for n in region if is_matmul(n)]
    (relu,) = [n for n in region if n.target is torch.ops.aten.relu.default]
    assert relu.args[0] is mm
    assert list(region.outputs) == [relu]  # relu still feeds mm_1 outside


def test_consumer_mm_keeps_a_prologue_from_another_partition():
    found = normal_regions(
        lambda x, w1, w2: torch.relu(x @ w1) @ torch.sin(w2), R(M, 32), R(32, 48), R(48, 16),
    )
    by_category = {r.category: r for r in found}
    assert sorted(by_category) == ["epilogue", "prologue"]
    assert any(n.target is torch.ops.aten.relu.default for n in by_category["epilogue"])
    prologue = by_category["prologue"]
    (sin,) = [n for n in prologue if n.target is torch.ops.aten.sin.default]
    (mm_1,) = [n for n in prologue if is_matmul(n)]
    assert mm_1.args[1] is sin
    assert not set(by_category["epilogue"]) & set(prologue)


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


# Ops without torch.Tag.pointwise that Inductor still lowers to Pointwise IR.
NO_POINTWISE_TAG_PROLOGUES = {
    "index": ("aten::index", lambda t, i, w: t[i] @ w),
    "embedding": ("aten::embedding", lambda t, i, w: torch.nn.functional.embedding(i, t) @ w),
    "gather": ("aten::gather", lambda t, i, w: torch.gather(t[:M], 0, i[:, None].expand(M, 32)) @ w),
    "full": ("aten::full", lambda t, i, w: (t[:M] + torch.full((M, 32), 2.0)) @ w),
    "iota": ("prims::iota", lambda t, i, w: (t[:M] + torch.ops.prims.iota(
        32, start=0, step=1, dtype=torch.int64, device=t.device, requires_grad=False)) @ w),
    "split": ("aten::split", lambda t, i, w: torch.split(t[:M] * 2, 16, dim=1)[0] @ w[:16]),
    "index_put": ("aten::index_put", lambda t, i, w: t.index_put((i,), torch.ones(M, 32))[:M] @ w),
}


@pytest.mark.parametrize("name", NO_POINTWISE_TAG_PROLOGUES)
def test_partition_without_reduction_or_scan_is_a_prologue(name):
    op, fn = NO_POINTWISE_TAG_PROLOGUES[name]
    idx = torch.randint(0, 100 if name != "gather" else M, (M,))
    (region,) = normal_regions(fn, R(100, 32), idx, R(32, 48))
    assert region.category == "prologue"
    assert any(op_name(n) == op for n in region)


# Partition members that combine a whole axis or accumulate, without Tag.reduction.
REDUCTION_OR_SCAN_PROLOGUES = {
    "sort": ("aten::sort", lambda x, i, w: torch.sort(x, dim=1).values @ w),
    "index_put_accumulate": (
        "aten::index_put", lambda x, i, w: x.index_put((i,), torch.ones(M, 32), accumulate=True) @ w,
    ),
    "scatter_add": (
        "aten::scatter_add", lambda x, i, w: x.scatter_add(0, i[:, None].expand(M, 32), torch.ones(M, 32)) @ w,
    ),
}


@pytest.mark.parametrize("name", REDUCTION_OR_SCAN_PROLOGUES)
def test_partition_with_untagged_reduction_or_scan_is_not_a_prologue(name):
    op, fn = REDUCTION_OR_SCAN_PROLOGUES[name]
    gm = make_fx(fn, tracing_mode="fake")(R(M, 32), torch.randint(0, M, (M,)), R(32, 48))
    assert any(op_name(n) == op for p in get_fusion_partitions(gm) for n in p)
    assert get_matmul_regions(gm, partition_node_is_supported, categories=("prologue", "epilogue")) == []


def _call(target, *args, **kwargs):
    graph = torch.fx.Graph()
    placeholders = [graph.placeholder(f"a{i}") if a is None else a for i, a in enumerate(args)]
    return graph.call_function(target, tuple(placeholders), kwargs)


aten, prims = torch.ops.aten, torch.ops.prims


@pytest.mark.parametrize(
    "node, expected",
    [
        # Missing torch.Tag.reduction, still a reduction or scan.
        (_call(aten.cumsum.default, None, 1), True),
        (_call(aten._softmax.default, None, -1, False), True),
        (_call(aten.native_layer_norm.default, None, [8], None, None, 1e-5), True),
        (_call(prims.sum.default, None, [1]), True),
        (_call(aten.sort.default, None, 1), True),
        (_call(aten.avg_pool2d.default, None, [2, 2]), True),
        (_call(aten.embedding_dense_backward.default, None, None, 10, -1, False), True),
        # Accumulation depends on an argument.
        (_call(aten.index_put.default, None, [None], None, True), True),
        (_call(aten.index_put.default, None, [None], None), False),
        (_call(aten.index_put.default, None, [None], None, accumulate=True), True),
        (_call(aten.scatter.reduce, None, 0, None, None, reduce="add"), True),
        (_call(aten.scatter.src, None, 0, None, None), False),
        # Tagged reductions, and pointwise overloads of the same packet.
        (_call(aten.max.dim, None, 1), True),
        (_call(aten.max.other, None, None), False),
        (_call(aten.var_mean.correction, None, [1]), True),
        # Gathers, constants and views have no Tag.pointwise but are not reductions.
        (_call(aten.embedding.default, None, None), False),
        (_call(aten.index.Tensor, None, [None]), False),
        (_call(aten.full.default, [4], 1.0), False),
        (_call(prims.iota.default, 4, start=0, step=1, dtype=torch.int64,
               device=torch.device("cpu"), requires_grad=False), False),
        (_call(aten.split.Tensor, None, 2), False),
        (_call(aten._unsafe_view.default, None, [4]), False),
        (_call(aten.mm.default, None, None), False),
    ],
    ids=lambda v: str(v.target) if isinstance(v, torch.fx.Node) else str(v),
)
def test_is_reduction_or_scan(node, expected):
    assert is_reduction_or_scan(node) is expected


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
    assert get_matmul_regions(gm, partition_node_is_supported, categories=("prologue",)) == []


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


