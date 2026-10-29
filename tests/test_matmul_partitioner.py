"""matmul_partitioner, grouped by the rules in its module comment."""

from __future__ import annotations

import json

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import (
    _CaptureState,
    compute_partition_pattern,
    extract_partition_subgraph,
    get_fusion_partitions,
    partition_node_is_supported,
)
from matmul_partitioner import (
    MatmulLimits,
    MatmulRegion,
    get_matmul_regions,
    is_matmul,
    is_reduction,
)


def trace(fn, *shapes):
    return make_fx(fn, tracing_mode="fake")(*[torch.randn(*s) for s in shapes])


def regions_of(gm):
    return get_fusion_partitions(gm, only_include_mm=True)


def dots(region):
    return {n for n in region if is_matmul(n)}


def row_norm(x):
    centered = x - x.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(variance + 1e-5)


# --- Rule 1: one reduction-chain budget for MMs and ordinary reductions ---


def test_chain_stops_before_the_sixth_reduction():
    gm = trace(
        lambda x, w: (
            (x @ w)
            .view(4, 4, 4, 4, 4)
            .sum(4)
            .relu()
            .sum(3)
            .relu()
            .sum(2)
            .relu()
            .sum(1)
            .sigmoid()
            .sum(0)
        ),
        (4, 1024),
        (1024, 256),
    )
    region = regions_of(gm)[0]
    assert region.metrics["contraction_depth"] == 5
    assert region.outputs[0].target == torch.ops.aten.sigmoid.default


def test_gather_prenorm_mm_residual_postnorm_is_one_region():
    def fn(table, indices, w, residual):
        return row_norm(row_norm(table.index_select(0, indices)) @ w + residual)

    inputs = (
        torch.randn(64, 128),
        torch.arange(8),
        torch.randn(128, 128),
        torch.randn(8, 128),
    )
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    full = next(r for r in regions_of(gm) if r.metrics["contraction_depth"] == 5)
    assert full.metrics["mm"] == 1
    assert full.metrics["reduction_depth"] == 4
    assert any(n.target == torch.ops.aten.index_select.default for n in full)
    sub, _, _ = extract_partition_subgraph(full, gm)
    torch.testing.assert_close(sub(*inputs), fn(*inputs))


def test_later_reduction_does_not_hide_the_norm_mm_norm_window():
    def fn(table, indices, w, residual):
        return row_norm(row_norm(table.index_select(0, indices)) @ w + residual).sum(-1)

    gm = make_fx(fn, tracing_mode="fake")(
        torch.randn(64, 128),
        torch.arange(8),
        torch.randn(128, 128),
        torch.randn(8, 128),
    )
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    mm = next(n for n in gm.graph.nodes if is_matmul(n))
    regions = regions_of(gm)
    assert any(
        sum(is_reduction(n) for n in r if order[n] < order[mm]) == 2
        and sum(is_reduction(n) for n in r if order[n] > order[mm]) == 2
        and r.outputs[0].target == torch.ops.aten.mul.Tensor
        for r in regions
    )
    assert all(r.metrics["contraction_depth"] <= 5 for r in regions)


# --- Rule 2: reduction sizes ---


@pytest.mark.parametrize(
    "shapes,chained",
    [
        (((8, 8192), (8192, 64), (64, 64)), True),  # first MM's K is free
        (((8, 64), (64, 128), (128, 4096)), True),  # and so is any N
        (((8, 64), (64, 129), (129, 64)), False),  # a chained MM's K is not
    ],
)
def test_only_the_k_of_a_chained_mm_is_bounded(shapes, chained):
    regions = regions_of(trace(lambda x, a, b: ((x @ a).relu() @ b).relu(), *shapes))
    assert regions
    assert any(len(dots(r)) == 2 for r in regions) == chained


@pytest.mark.parametrize("kind", ["mm", "bmm", "addmm", "baddbmm"])
@pytest.mark.parametrize("k", [128, 129])
def test_every_mm_variant_uses_the_same_k_limit(kind, k):
    if kind in {"bmm", "baddbmm"}:
        shapes = (2, 8, 1024), (2, 1024, k), (2, k, 256)

        def fn(x, a, b):
            h = torch.bmm(x, a).relu()
            if kind == "baddbmm":
                return torch.baddbmm(torch.zeros(2, 8, 256), h, b).sigmoid()
            return torch.bmm(h, b).sigmoid()

    else:
        shapes = (8, 1024), (1024, k), (k, 256)

        def fn(x, a, b):
            h = (x @ a).relu()
            if kind == "addmm":
                return torch.addmm(torch.zeros(256), h, b).sigmoid()
            return (h @ b).sigmoid()

    regions = regions_of(trace(fn, *shapes))
    assert any(r.metrics["mm"] == 2 for r in regions) == (k == 128)


@pytest.mark.parametrize("n", [128, 129])
def test_reduction_after_mm_is_bounded_by_its_size(n):
    gm = trace(lambda x, w: (x @ w).relu().sum(1).sigmoid(), (8, 1024), (1024, n))
    (region,) = regions_of(gm)
    kept = n <= 128
    assert region.metrics["reduction_depth"] == int(kept)
    expected = torch.ops.aten.sigmoid if kept else torch.ops.aten.relu
    assert region.outputs[0].target == expected.default


@pytest.mark.parametrize("extent", [128, 129])
def test_reduction_before_mm_is_bounded_by_its_size(extent):
    gm = trace(
        lambda x, w: x.abs().sum(1, keepdim=True).relu() @ w, (8, extent), (1, 64)
    )
    (region,) = regions_of(gm)
    targets = {n.target for n in region}
    assert torch.ops.aten.relu.default in targets
    assert (torch.ops.aten.abs.default in targets) == (extent == 128)
    assert region.metrics["contraction_depth"] == (2 if extent == 128 else 1)


@pytest.mark.parametrize("widths", [(16, 8), (16, 16)])
def test_multi_axis_reduction_size_is_the_product_of_its_axes(widths):
    a, b = widths
    gm = trace(
        lambda x, w: (x @ w).view(8, a, b).relu().sum((1, 2)), (8, 1024), (1024, a * b)
    )
    region = regions_of(gm)[0]
    assert region.metrics["reduction_depth"] == int(a * b <= 128)


def test_tuple_valued_reduction_inside_a_region():
    gm = trace(lambda x, w: (x @ w).relu().max(dim=1).values.sigmoid(), (8, 512), (512, 128))
    (region,) = regions_of(gm)
    assert region.metrics["reduction_depth"] == 1
    assert region.outputs[0].target == torch.ops.aten.sigmoid.default


def test_oversized_tuple_reduction_becomes_tensor_inputs():
    def fn(x, w):
        variance, mean = torch.var_mean(x, dim=1, correction=0, keepdim=True)
        return ((x - mean) * torch.rsqrt(variance + 1e-5)) @ w

    x, w = torch.randn(8, 256), torch.randn(256, 64)
    gm = make_fx(fn, tracing_mode="fake")(x, w)
    region = regions_of(gm)[0]
    # var_mean reduces 256 > 128 elements: a boundary, not part of the region.
    assert all(is_matmul(n) for n in region if is_reduction(n))
    pattern = compute_partition_pattern(region, gm)
    sub = pattern["sub_gm"]
    assert not any(n.op == "get_attr" for n in sub.graph.nodes)
    assert len(pattern["placeholder_info"]) == 4  # x, w, variance, mean
    variance, mean = torch.var_mean(x, dim=1, correction=0, keepdim=True)
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    values = {n.name: value for n, value in zip(placeholders, (x, w))}
    reduction = next(n for n in gm.graph.nodes if is_reduction(n) and not is_matmul(n))
    for n in reduction.users:
        values[n.name] = (variance, mean)[n.args[1]]
    extracted, _, _ = extract_partition_subgraph(region, gm)
    args = [values[n.name] for n in extracted.graph.nodes if n.op == "placeholder"]
    torch.testing.assert_close(sub(*args), fn(x, w))


# --- Rule 3 and the horizontal pass ---


@pytest.mark.parametrize("shared_input", [True, False])
def test_independent_branches_form_one_horizontal_region(shared_input):
    def fn(*args):
        return tuple(
            (args[0 if shared_input else 2 * i] @ args[2 * i + 1]).relu()
            for i in range(3)
        )

    (region,) = regions_of(trace(fn, *((8, 1024), (1024, 256)) * 3))
    assert region.metrics["mm"] == region.metrics["outputs"] == 3
    assert region.metrics["contraction_depth"] == 1


@pytest.mark.parametrize("width", [64, 32])
def test_horizontal_mms_need_equal_output_shapes_but_k_may_differ(width):
    gm = trace(
        lambda x, a, y, b: ((x @ a).relu(), (y @ b).relu()),
        (8, 256),
        (256, 64),
        (8, 512),
        (512, width),
    )
    expected = [2] if width == 64 else [1, 1]
    assert sorted(r.metrics["mm"] for r in regions_of(gm)) == expected


def test_join_cannot_combine_mms_with_different_output_shapes():
    # (8, 64) + (8, 1) broadcasts fine, but rule 3 still keeps the MMs apart.
    gm = trace(
        lambda x, a, y, b: (x @ a).relu() + y @ b,
        (8, 256),
        (256, 64),
        (8, 512),
        (512, 1),
    )
    regions = regions_of(gm)
    assert regions and all(r.metrics["mm"] == 1 for r in regions)


def test_depth_is_counted_per_path_not_per_region():
    gm = trace(
        lambda x, a, b, c, d: (
            ((x @ a).relu() @ b).relu(),
            ((x @ c).relu() @ d).relu(),
        ),
        (8, 1024),
        (1024, 128),
        (128, 128),
        (1024, 128),
        (128, 128),
    )
    (region,) = regions_of(gm)
    assert region.metrics["mm"] == 4
    assert region.metrics["contraction_depth"] == 2


def test_swiglu_gate_is_kept_and_the_large_down_projection_is_cut():
    def fn(x, a, b, c):
        return (torch.nn.functional.silu(x @ a) * (x @ b)) @ c

    gm = trace(fn, (8, 1024), (1024, 3072), (1024, 3072), (3072, 1024))
    mm = [n for n in gm.graph.nodes if is_matmul(n)]
    regions = regions_of(gm)
    gate = next(r for r in regions if dots(r) == set(mm[:2]))
    assert any(n.target == torch.ops.aten.mul.Tensor for n in gate.outputs)
    assert not any(set(mm) <= set(r) for r in regions)


# --- Rule 4: outputs need an MM and real computation ---


@pytest.mark.parametrize("kind", ["bare", "cast", "views", "copy", "identity"])
def test_mm_with_only_views_casts_or_copies_is_not_a_region(kind):
    def fn(x, w):
        h = x @ w
        if kind == "cast":
            return h.double()
        if kind == "views":
            return h.t().t()
        if kind == "copy":
            return h.clone()
        if kind == "identity":
            return h * 1 + 0
        return h

    assert regions_of(trace(fn, (8, 16), (16, 32))) == []


def test_only_useful_outputs_are_returned_and_the_model_is_untouched():
    def fn(x, w, v):
        h = x @ w
        return h, h.relu(), (x @ v).double()

    x, w, v = torch.randn(8, 16), torch.randn(16, 32), torch.randn(16, 32)
    gm = make_fx(fn, tracing_mode="fake")(x, w, v)
    original = str(gm.graph)
    (region,) = regions_of(gm)
    assert isinstance(region, MatmulRegion)
    assert [n.target for n in region.outputs] == [torch.ops.aten.relu.default]
    assert str(gm.graph) == original
    sub, _, _ = extract_partition_subgraph(region, gm)
    torch.testing.assert_close(sub(x, w), (x @ w).relu())


@pytest.mark.parametrize("prologue", ["relu", "gather"])
def test_real_prologue_makes_a_bare_mm_useful(prologue):
    def fn(x, w, idx):
        x = x.relu() if prologue == "relu" else x.index_select(0, idx)
        return x @ w

    gm = make_fx(fn, tracing_mode="fake")(
        torch.randn(64, 1024), torch.randn(1024, 256), torch.zeros(8, dtype=torch.int64)
    )
    (region,) = regions_of(gm)
    assert region.outputs[0].target == torch.ops.aten.mm.default
    assert region.metrics["non_mm_compute"] == 1


def test_bare_horizontal_mms_are_useful_together():
    same = trace(lambda x, a, b: (x @ a, x @ b), (8, 256), (256, 64), (256, 64))
    (region,) = regions_of(same)
    assert region.metrics["mm"] == 2 and region.metrics["non_mm_compute"] == 0
    different = trace(lambda x, a, b: (x @ a, x @ b), (8, 256), (256, 64), (256, 32))
    assert regions_of(different) == []


# --- Rule 5: low-value single-MM regions ---


@pytest.mark.parametrize("kind", ["padding", "bias_packing", "copy", "infinity_mask"])
def test_low_value_regions_are_dropped(kind):
    if kind == "padding":
        gm = trace(
            lambda x, w: (x @ torch.nn.functional.pad(w, (0, 2)))[:, :-2], (4, 8), (8, 8)
        )
    elif kind == "bias_packing":
        gm = trace(
            lambda x, w, b, c: torch.addmm(torch.cat([b, c]), x, w),
            (4, 8),
            (8, 8),
            (4,),
            (4,),
        )
    elif kind == "copy":
        gm = trace(
            lambda x, w: (
                torch.ops.aten.select_scatter.default(x.new_zeros(4, 1, 8), x, 1, 0).view(
                    4, 8
                )
                @ w
            ),
            (4, 8),
            (8, 8),
        )
    else:
        gm = trace(lambda x, w: torch.bmm(x, w) == -float("inf"), (2, 4, 8), (2, 8, 8))
    diagnostics = {}
    regions = get_matmul_regions(gm, partition_node_is_supported, diagnostics=diagnostics)
    assert regions == []
    assert diagnostics["rejections"]["no_substantive_fusion"] > 0


def test_real_work_on_a_padded_mm_is_kept():
    gm = trace(
        lambda x, w: (x @ torch.nn.functional.pad(w, (0, 2)))[:, :-2].relu(),
        (4, 8),
        (8, 8),
    )
    regions = regions_of(gm)
    assert any(n.target == torch.ops.aten.relu.default for r in regions for n in r)


# --- Search details ---


def test_pointwise_between_mms_belongs_to_the_producing_mm():
    # The second MM's K (1024) is too large to chain, so the relu is the first
    # MM's epilogue and is not repeated as the second MM's prologue.
    gm = trace(lambda x, a, b: (x @ a).relu() @ b, (8, 512), (512, 1024), (1024, 128))
    mm = [n for n in gm.graph.nodes if is_matmul(n)]
    (region,) = regions_of(gm)
    assert dots(region) == {mm[0]}
    assert region.outputs[0].target == torch.ops.aten.relu.default


def test_too_many_outputs_fall_back_to_a_smaller_window():
    def fn(x, w):
        h = (x @ w).relu()
        return tuple(h + i for i in range(1, 7))

    regions = regions_of(trace(fn, (8, 128), (128, 128)))
    assert regions
    assert all(len(r.outputs) <= MatmulLimits().max_outputs for r in regions)


# --- Rule 6: standalone replay ---


def test_path_that_leaves_and_reenters_the_region_is_cut():
    def fn(x, w):
        h = x @ w
        return h.relu() + h.sin()

    gm = trace(fn, (8, 16), (16, 32))
    excluded = {n for n in gm.graph.nodes if n.target == torch.ops.aten.sin.default}
    regions = get_fusion_partitions(gm, excluded=excluded, only_include_mm=True)
    assert regions
    assert all(n.target != torch.ops.aten.add.Tensor for r in regions for n in r)
    assert regions[0].outputs[0].target == torch.ops.aten.relu.default


def test_unbind_is_a_boundary_even_if_the_callback_accepts_it():
    def fn(x, a, b):
        first, second = (x @ a).relu().view(8, 2, 64).unbind(1)
        return (first @ b).relu(), second

    gm = trace(fn, (8, 256), (256, 128), (64, 32))
    mm = [n for n in gm.graph.nodes if is_matmul(n)]
    regions = get_matmul_regions(gm, lambda n: True)
    assert all(len(dots(r)) == 1 for r in regions)
    assert all(n.target != torch.ops.aten.unbind.int for r in regions for n in r)
    downstream = next(r for r in regions if mm[1] in r)
    pattern = compute_partition_pattern(downstream, gm)
    placeholders = [n for n in pattern["sub_gm"].graph.nodes if n.op == "placeholder"]
    assert len(placeholders) == 2
    assert all(isinstance(n.meta["val"], torch.Tensor) for n in placeholders)
    x, a, b = torch.randn(8, 256), torch.randn(256, 128), torch.randn(64, 32)
    first = (x @ a).relu().view(8, 2, 64).unbind(1)[0]
    torch.testing.assert_close(pattern["sub_gm"](first, b), (first @ b).relu())


# --- Capture integration ---


def test_capture_writes_matmul_entries_and_merges_into_mm_directories(tmp_path):
    from merge_captures import merge_one_capture

    gm = trace(lambda x, w: ((x @ w).relu(), (x @ w).double()), (8, 16), (16, 32))
    capture = tmp_path / "capture"
    state = _CaptureState(
        str(capture),
        graph_dir=str(tmp_path / "full"),
        validate=False,
        only_include_mm=True,
    )
    state.process_graph(gm)
    state.finalize()
    index = json.loads((capture / "index.json").read_text())
    assert index["n_dropped"] == 0 and index["n_captured"] == 1
    entry = index["captured"][0]
    assert entry["kind"] == "matmul" and entry["mm_types"] == ["mm"]
    assert entry["matmul"]["outputs"] == 1
    sidecar = json.loads((tmp_path / "full" / "full_graph_000.meta.json").read_text())
    assert sidecar["roundtrip"] == "ok"
    merge_one_capture(capture, tmp_path / "corpus", "case", suite="test", mode="infer")
    assert len(list((tmp_path / "corpus" / "canonical").glob("mm_*/repro.py"))) == 1
