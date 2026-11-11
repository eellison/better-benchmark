"""matmul_partitioner: the examples of each pattern category."""

from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import get_fusion_partitions, partition_node_is_supported
from matmul_partitioner import (
    MatmulLimits,
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
    # The transformed operand is also a model output, but it does not depend
    # on the MM, so trimming returns only the MM's result.
    assert list(region.outputs) == [mm]


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


def test_k_row_norm_partition_is_a_prologue():
    # One reduction normalizing rows of A along K: LayerNorm(x) @ w.
    (region,) = normal_regions(lambda x, w: ln(x) @ w, R(M, 32), R(32, 48))
    assert region.category == "prologue"
    assert sum(n.target is torch.ops.aten.var_mean.correction for n in region) == 1


def test_per_head_k_norm_partition_is_a_prologue():
    # Per-head RMS norm: runs of 8 that tile K = 32.
    def fn(x, w):
        h = x.view(M, 4, 8)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return h.view(M, 32) @ w

    (region,) = normal_regions(fn, R(M, 32), R(32, 48))
    assert region.category == "prologue"


@pytest.mark.parametrize("name, fn", [
    # Reduces over M (a column statistic), not along K.
    ("column", lambda x, w: (x / x.sum(0, keepdim=True)) @ w),
    # Two reductions in one partition.
    ("two", lambda x, w: ((x - x.amax(-1, keepdim=True)) / x.sum(-1, keepdim=True)) @ w),
    # Normalizes B along its last axis (N), not A along K.
    ("weight", lambda x, w: x @ (w / w.sum(-1, keepdim=True))),
])
def test_other_reduction_partitions_are_not_prologues(name, fn):
    assert normal_regions(fn, R(M, 32), R(32, 48)) == []


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
    # h escapes to the other MM and the model output, but it does not depend
    # on the region's MM, so trimming returns only MM-dependent values.
    assert all(len(r.outputs) == 1 for r in found)


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


def horizontal_regions(fn, *inputs, limits=None):
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    found = get_matmul_regions(
        gm, partition_node_is_supported, limits=limits, categories=("horizontal", "horizontal_combine")
    )
    return gm, found


def extracted_category(gm, region):
    from capture_hook import extract_partition_subgraph

    subset, _, _ = extract_partition_subgraph(region, gm)
    return classify_pattern_graph(subset)


def test_horizontal_shares_an_operand_in_either_slot():
    # x @ a, x @ b concatenate along N; a @ x, b @ x along M.
    for fn in (lambda x, a, b: (x @ a, x @ b), lambda x, a, b: (a @ x, b @ x)):
        gm, found = horizontal_regions(fn, *[R(32, 32) for _ in range(3)])
        assert [r.category for r in found] == ["horizontal"]
        assert extracted_category(gm, found[0]) == "horizontal"


# Same root and equal output shapes, but no shared operand: none concatenates
# into one GEMM.
SAME_ROOT_DIFFERENT_OPERANDS = {
    # Distinct tensors: make_fx gives repeated tensor objects one placeholder.
    "opposite_slots": (lambda x, a, b: (x @ a, b @ x), [R(32, 32) for _ in range(3)]),
    "transpose": (lambda x, a, b: (x @ a, x.t() @ b), [R(32, 32) for _ in range(3)]),
    "mixed_dtype": (
        lambda x, a, b: (x.half() @ a, x.bfloat16() @ b),
        [R(M, 32), R(32, 48).half(), R(32, 48).bfloat16()],
    ),
}


@pytest.mark.parametrize("name", SAME_ROOT_DIFFERENT_OPERANDS)
def test_same_root_is_not_a_shared_operand(name):
    fn, args = SAME_ROOT_DIFFERENT_OPERANDS[name]
    assert horizontal_regions(fn, *args)[1] == []


# Separate nodes with equal values are one shared operand.
EQUAL_OPERAND_NODES = {
    "two_casts": (
        lambda x, a, b: (x.half() @ a, x.half() @ b),
        [R(M, 32), R(32, 48).half(), R(32, 48).half()],
    ),
    "two_views": (
        lambda x, a, b: (x.view(M, 32) @ a, x.reshape(M, 32) @ b),
        [R(2, M // 2, 32), R(32, 48), R(32, 48)],
    ),
    "double_transpose": (lambda x, a, b: (x @ a, x.t().t() @ b), [R(M, 32), R(32, 48), R(32, 48)]),
}


@pytest.mark.parametrize("name", EQUAL_OPERAND_NODES)
def test_separate_nodes_with_equal_values_share_the_operand(name):
    fn, args = EQUAL_OPERAND_NODES[name]
    gm, found = horizontal_regions(fn, *args)
    assert [r.category for r in found] == ["horizontal"]
    assert extracted_category(gm, found[0]) == "horizontal"


def test_window_keeps_the_shared_operand_path_computed_from_an_mm():
    # The views of x depend on an earlier MM, so grow_backward alone leaves
    # them out and the extracted MMs would read view_1 and view_2 separately.
    def fn(h, w0, a, b):
        x = (h @ w0).relu()
        return x.view(M, 32) @ a, x.view(M, 32) @ b

    gm, found = horizontal_regions(fn, R(M, 32), R(32, 32), R(32, 48), R(32, 48))
    assert [r.category for r in found] == ["horizontal"]
    assert sum(op_name(n) == "aten::view" for n in found[0]) == 2
    assert extracted_category(gm, found[0]) == "horizontal"


def test_one_operand_node_feeding_all_mms_stays_an_input():
    def fn(h, w0, a, b):
        v = (h @ w0).relu().view(M, 32)
        return v @ a, v @ b

    gm, found = horizontal_regions(fn, R(M, 32), R(32, 32), R(32, 48), R(32, 48))
    assert [r.category for r in found] == ["horizontal"]
    assert not any(op_name(n) == "aten::view" for n in found[0])
    assert extracted_category(gm, found[0]) == "horizontal"


def test_horizontal_is_dropped_when_the_window_cannot_keep_the_sharing():
    fn = lambda x, a, b: (x.view(M, 32) @ a, x.view(M, 32) @ b)
    args = (R(2, M // 2, 32), R(32, 48), R(32, 48))
    assert [r.category for r in horizontal_regions(fn, *args)[1]] == ["horizontal"]
    # Two MMs and two views do not fit in three nodes; one view alone would
    # leave the MMs reading separate inputs.
    assert horizontal_regions(fn, *args, limits=MatmulLimits(max_nodes=3))[1] == []


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


def test_classify_requires_horizontal_mms_to_share_an_operand():
    shared = make_fx(lambda x, a, b: (x @ a, x @ b), tracing_mode="fake")(R(M, 32), R(32, 48), R(32, 48))
    unshared = make_fx(lambda x, y, a, b: (x @ a, y @ b), tracing_mode="fake")(
        R(M, 32), R(M, 32), R(32, 48), R(32, 48)
    )
    assert classify_pattern_graph(shared) == "horizontal"
    assert classify_pattern_graph(unshared) is None


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




def test_epilogue_outputs_that_do_not_depend_on_the_mm_are_trimmed():
    # The mask is fused into the epilogue that reads it, and the model also
    # returns it; storing the mask does not depend on the MM, so only the
    # where() result stays an output (the mask stays as an internal value).
    def fn(x, w, s):
        mask = torch.sigmoid(s) > 0.5
        return torch.where(mask, x @ w, 0.0), mask

    gm = make_fx(fn, tracing_mode="fake")(R(M, 32), R(32, 48), R(M, 48))
    (region,) = get_matmul_regions(gm, partition_node_is_supported, categories=("epilogue",))
    assert any(n.target is torch.ops.aten.gt.Scalar for n in region)
    assert [n.target for n in region.outputs] == [torch.ops.aten.where.self]


def test_normal_region_compute_is_bounded():
    from matmul_partitioner import MatmulLimits

    def fn(x, w):
        h = x @ w
        for _ in range(MatmulLimits().max_normal_compute + 1):
            h = torch.sin(h)
        return h

    assert normal_regions(fn, R(M, 32), R(32, 48)) == []


@pytest.mark.parametrize("extra, kept", [(4, True), (5, False)])
def test_epilogue_extra_inputs_are_bounded(extra, kept):
    # Extra inputs exclude the MM operands (here x is viewed before the MM).
    def fn(x, w, *side):
        h = x.view(M, 32) @ w
        for s in side:
            h = h * s
        return h

    side = [R(M, 48) for _ in range(extra)]
    gm = make_fx(fn, tracing_mode="fake")(R(M // 2, 2, 32), R(32, 48), *side)
    regions = get_matmul_regions(gm, partition_node_is_supported, categories=("epilogue",))
    assert bool(regions) is kept
    if kept:
        assert regions[0].metrics["extra_inputs"] == extra


def test_dropout_seed_buffer_is_not_an_extra_input():
    import torch._inductor.inductor_prims  # noqa: F401
    from matmul_partitioner import extra_inputs

    def fn(x, w, seeds):
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 0)
        keep = torch.ops.prims.inductor_random.default([M, 48], seed, "rand") > 0.1
        return (x @ w) * keep

    gm = make_fx(fn, tracing_mode="fake")(R(M, 32), R(32, 48), torch.zeros(2, dtype=torch.int64))
    (region,) = get_matmul_regions(gm, partition_node_is_supported, categories=("epilogue",))
    mm = next(n for n in region if is_matmul(n))
    assert extra_inputs(list(region), mm) == set()


# --- outputs: values, not views of them -------------------------------------


def epilogue_outputs(fn, *inputs):
    """Targets of the extracted epilogue's outputs, and its shape params."""
    from capture_hook import extract_partition_subgraph

    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    (region,) = get_matmul_regions(gm, partition_node_is_supported, categories=("epilogue",))
    subset, _, shape_params = extract_partition_subgraph(region, gm)
    output = next(n for n in subset.graph.nodes if n.op == "output")
    returned = output.args[0] if isinstance(output.args[0], tuple) else (output.args[0],)
    return [n.target for n in returned], shape_params


aten = torch.ops.aten


def test_views_of_one_value_are_returned_as_that_value_once():
    # Returned as (relu, view, t, expand) before; the views also added
    # _shape_param inputs.
    def fn(x, w):
        h = torch.relu(x @ w)
        return h, h.view(2, M // 2, 48), h.t(), h.unsqueeze(0).expand(3, M, 48)

    assert epilogue_outputs(fn, R(M, 32), R(32, 48)) == ([aten.relu.default], {})


def test_views_that_keep_every_element_include_full_slices_and_size_1_selects():
    def fn(x, w):
        h = torch.relu(x @ w).view(1, M, 48)
        return aten.select.int(h, 0, 0), aten.slice.Tensor(h, 1, 0, M)

    assert epilogue_outputs(fn, R(M, 32), R(32, 48))[0] == [aten.relu.default]


def test_views_that_drop_elements_stay_unless_they_view_another_output():
    # DeiT's two class tokens: returning the whole relu would claim every token.
    def tokens(x, w):
        h = torch.relu(x @ w).view(2, M // 2, 48)
        return h.select(1, 0), h.select(1, 1)

    assert epilogue_outputs(tokens, R(M, 32), R(32, 48))[0] == [aten.select.int] * 2

    # BERT_pytorch also returns the whole value, so its token 0 adds nothing.
    def token_and_value(x, w):
        h = torch.relu(x @ w).view(2, M // 2, 48)
        return h.select(1, 0), h.view(M, 48)

    assert epilogue_outputs(token_and_value, R(M, 32), R(32, 48))[0] == [aten.relu.default]


def test_casts_and_clones_are_outputs_of_their_own():
    def fn(x, w):
        h = torch.relu(x @ w)
        cast = torch.ops.prims.convert_element_type.default(h, torch.bfloat16)
        transposed = h.view(2, M // 2, 4, 12).permute(0, 2, 1, 3).contiguous()
        return cast.view(2, M // 2, 48), transposed

    assert epilogue_outputs(fn, R(M, 32), R(32, 48))[0] == [
        torch.ops.prims.convert_element_type.default, aten.clone.default,
    ]


def test_horizontal_view_outputs_are_the_mm_results():
    # A ConvBert QKV window returned (addmm_2, view_3, view_4, view_5).
    def fn(x, a, b, c):
        q, k, v = x @ a, x @ b, x @ c
        return v, v.view(2, M // 2, 48), q.view(2, M // 2, 4, 12), k.view(2, M // 2, 4, 12)

    gm, found = horizontal_regions(fn, R(M, 32), *[R(32, 48)] * 3)
    assert [r.category for r in found] == ["horizontal"]
    assert len(found[0].outputs) == 3 and all(map(is_matmul, found[0].outputs))
    assert not any(op_name(n) == "aten::view" for n in found[0])


# --- casts, copies and views alone are bare MMs; CPU inputs are left out ------


def test_cast_copy_or_view_only_epilogues_are_bare_mms():
    # 0 < non-MM compute: casts, copies and views are not computation.
    cast = lambda x, w: torch.ops.prims.convert_element_type.default(x @ w, torch.float32)
    copy = lambda x, w: (x @ w).view(2, M // 2, 4, 12).permute(0, 2, 1, 3).contiguous()
    view = lambda x, w: (x @ w).view(2, M // 2, 4, 12).permute(0, 2, 1, 3)
    args = (R(M, 32).bfloat16(), R(32, 48).bfloat16())
    for fn in (cast, copy, view):
        assert normal_regions(fn, *args) == []
    # With real computation the cast stays in the epilogue.
    (region,) = normal_regions(
        lambda x, w: torch.ops.prims.convert_element_type.default(torch.relu(x @ w), torch.float32),
        *args,
    )
    assert region.category == "epilogue"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU device")
def test_regions_that_read_a_cpu_tensor_are_left_out():
    from matmul_partitioner import cpu_inputs

    # DeBERTa-v2 divides attention scores by a 0-d CPU tensor.
    epilogue = lambda x, w, scale: (x @ w) / scale
    x, a, b = R(M, 32, device="cuda"), R(32, 48, device="cuda"), R(32, 48, device="cuda")
    assert normal_regions(epilogue, x, a, torch.tensor(8.0, device="cuda"))
    assert normal_regions(epilogue, x, a, torch.tensor(8.0)) == []

    # A horizontal window falls back to the window without the division.
    pair = lambda x, a, b, scale: ((x @ a) / scale, (x @ b) / scale)
    _, found = horizontal_regions(pair, x, a, b, torch.tensor(8.0))
    assert [r.category for r in found] == ["horizontal"]
    assert not cpu_inputs(found[0])
    assert not any(n.target is aten.div.Tensor for n in found[0])
