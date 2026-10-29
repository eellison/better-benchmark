from __future__ import annotations

import json
import operator
from dataclasses import asdict

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import (
    _CaptureState,
    extract_partition_subgraph,
    get_fusion_partitions,
)
from matmul_partitioner import MatmulLimits, MatmulRegion, is_matmul, is_reduction
from roundtrip_validation import derive_partitions


def trace(fn, *shapes):
    return make_fx(fn, tracing_mode="fake")(*[torch.randn(*s) for s in shapes])


def dots(region):
    return {n for n in region if is_matmul(n)}


@pytest.mark.parametrize("kind", ["bare", "cast", "views", "copy", "identity"])
def test_no_non_mm_computation_is_dropped(kind):
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
        if kind == "horizontal":
            return h, x @ w
        return h

    assert (
        get_fusion_partitions(trace(fn, (8, 16), (16, 32)), only_include_mm=True) == []
    )


def test_meaningful_outputs_only_and_unused_mm_branch_removed():
    gm = trace(
        lambda x, w, v: ((x @ w), (x @ w).relu(), (x @ v).double()),
        (8, 16),
        (16, 32),
        (16, 32),
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    region = regions[0]
    assert isinstance(region, MatmulRegion)
    assert len(dots(region)) == 1
    assert [n.target for n in region.outputs] == [torch.ops.aten.relu.default]
    sub, _, _ = extract_partition_subgraph(region, gm)
    out = next(n for n in sub.graph.nodes if n.op == "output")
    assert not isinstance(out.args[0], tuple)


def test_escaping_bare_mm_output_is_pruned_but_model_output_is_unchanged():
    def fn(x, w):
        h = x @ w
        return h, h.relu()

    gm = trace(fn, (8, 16), (16, 32))
    original = str(gm.graph)
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions[0].outputs) == 1
    assert regions[0].outputs[0].target == torch.ops.aten.relu.default
    assert str(gm.graph) == original


def test_real_prologue_makes_matmul_output_meaningful():
    gm = trace(lambda x, w: x.relu() @ w, (8, 16), (16, 32))
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert regions[0].outputs[0].target == torch.ops.aten.mm.default
    assert regions[0].metrics["non_mm_compute"] == 1


def test_inter_mm_pw_is_owned_by_the_producer():
    gm = trace(lambda x, a, b: (x @ a).relu() @ b, (8, 512), (512, 1024), (1024, 128))
    mm = [n for n in gm.graph.nodes if is_matmul(n)]
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert dots(regions[0]) == {mm[0]}
    assert regions[0].outputs[0].target == torch.ops.aten.relu.default


def test_genuine_gather_prologue_is_kept():
    gm = make_fx(lambda x, w, idx: x.index_select(0, idx) @ w, tracing_mode="fake")(
        torch.randn(64, 1024), torch.randn(1024, 256), torch.zeros(8, dtype=torch.int64)
    )
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    assert region.metrics["non_mm_compute"] == 1
    assert any(n.target == torch.ops.aten.index_select.default for n in region)


def test_capture_directory_uses_actual_mm_op_names(tmp_path):
    from merge_captures import merge_one_capture

    capture = tmp_path / "capture"
    state = _CaptureState(str(capture), only_include_mm=True, validate=False)
    gm = trace(lambda x, w: (x @ w).relu(), (8, 16), (16, 32))
    state.process_graph(gm)
    state.finalize()
    assert state.captured[0]["mm_types"] == ["mm"]
    corpus = tmp_path / "corpus"
    merge_one_capture(capture, corpus, "case", suite="test", mode="infer")
    assert len(list((corpus / "canonical").glob("mm_*/repro.py"))) == 1


def test_small_vertical_chain_and_large_free_rows():
    gm = trace(
        lambda x, a, b: ((x @ a).relu() @ b).tanh(), (4096, 64), (64, 64), (64, 64)
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert any(len(dots(r)) == 2 for r in regions)
    assert all(r.metrics["handoff"] <= 128 for r in regions)


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("shared", [True, False])
def test_independent_horizontal_branches(count, shared):
    def fn(*args):
        return tuple(
            (args[0 if shared else 2 * i] @ args[2 * i + 1]).relu()
            for i in range(count)
        )

    gm = trace(fn, *((8, 1024), (1024, 256)) * count)
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert regions[0].metrics["mm"] == count
    assert regions[0].metrics["outputs"] == count
    assert regions[0].metrics["contraction_depth"] == 1


@pytest.mark.parametrize("count", [2, 3, 4])
def test_bare_horizontal_mm_are_meaningful_together(count):
    gm = trace(
        lambda x, *weights: tuple(x @ w for w in weights),
        (8, 1024),
        *((1024, 256),) * count,
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert regions[0].metrics["mm"] == count
    assert regions[0].metrics["non_mm_compute"] == 0
    assert len(regions[0].outputs) == count


@pytest.mark.parametrize(
    "shapes,kept",
    [
        (((8, 8192), (8192, 64), (64, 64)), True),
        (((8, 64), (64, 129), (129, 64)), False),
        (((8, 64), (64, 128), (128, 4096)), True),
    ],
)
def test_only_later_contracted_dimensions_are_bounded(shapes, kept):
    gm = trace(lambda x, a, b: ((x @ a).relu() @ b).relu(), *shapes)
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert regions
    assert any(len(dots(r)) == 2 for r in regions) == kept


def test_mm_mv_allows_512_initial_k_and_small_handoff():
    gm = trace(lambda x, a, v: torch.mv((x @ a).relu(), v), (8, 512), (512, 64), (64,))
    assert any(
        len(dots(r)) == 2 for r in get_fusion_partitions(gm, only_include_mm=True)
    )


def test_long_row_axis_has_no_streaming_exception():
    gm = trace(
        lambda x, a, b: ((x @ a).relu().t() @ b).relu(),
        (1024, 64),
        (64, 64),
        (1024, 64),
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert regions
    assert all(len(dots(r)) == 1 for r in regions)


@pytest.mark.parametrize(
    "k,n,kept", [(256, 128, True), (256, 129, False), (8192, 64, True)]
)
def test_mm_row_reduction_envelope(k, n, kept):
    gm = trace(lambda x, w: (x @ w).sum(-1), (8, k), (k, n))
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert bool(regions) == kept
    if regions:
        assert regions[0].metrics["reduction_extent"] == n


@pytest.mark.parametrize("kind", ["sum", "max", "amax", "mean"])
def test_general_reduction_after_multiple_pointwise_ops(kind):
    def fn(x, w):
        h = (x @ w).relu().square() * 0.25
        if kind == "max":
            return h.max(dim=1).values.sigmoid()
        return getattr(h, kind)(dim=1).sigmoid()

    gm = trace(fn, (8, 512), (512, 128))
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert regions[0].metrics["reduction_depth"] == 1
    assert regions[0].metrics["reduction_extent"] == 128
    assert regions[0].outputs[0].target == torch.ops.aten.sigmoid.default


def test_block_reduction_uses_actual_axes():
    gm = trace(
        lambda x, w: (x @ w).relu().view(2, 4, 8, 16).sum((1, 3)), (8, 256), (256, 128)
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert regions[0].metrics["reduction_extent"] == 64


def test_repeated_ordinary_reductions_accumulate_extent():
    gm = trace(lambda x, w: (x @ w).view(8, 64, 64).sum(2).sum(1), (8, 64), (64, 4096))
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert regions
    assert len(regions) == 1
    assert regions[0].metrics["reduction_depth"] == 2
    assert regions[0].metrics["contraction_depth"] == 3
    assert regions[0].metrics["reduction_extent"] == 4096
    assert regions[0].metrics["max_chained_reduction_size"] == 64


def test_four_small_mm_do_not_multiply_reduction_extent():
    gm = trace(
        lambda x, a, b, c, d: ((((x @ a).relu() @ b).relu() @ c).relu() @ d).relu(),
        (4096, 64),
        *((64, 64),) * 4,
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert any(r.metrics["mm"] == 4 for r in regions)
    assert all(r.metrics["contraction_depth"] <= 5 for r in regions)
    assert all(r.metrics["reduction_extent"] == 1 for r in regions)


def test_mm_and_reduce_share_depth_limit():
    gm = trace(
        lambda x, a, b, c, d: ((((x @ a).relu() @ b).relu() @ c).relu() @ d).sum(1),
        (8, 64),
        *((64, 64),) * 4,
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert regions
    assert all(r.metrics["contraction_depth"] <= 5 for r in regions)
    assert any(
        r.metrics["mm"] == 4 and r.metrics["reduction_depth"] == 1 for r in regions
    )


def test_full_tensor_input_resets_broadcast_statistic_coverage():
    def fn(x, w, residual):
        stats = (x @ w).sum(1, keepdim=True)
        return (residual + stats).sum(1)

    gm = trace(fn, (8, 64), (64, 64), (8, 64))
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert regions[0].metrics["reduction_depth"] == 2
    assert regions[0].metrics["reduction_extent"] == 64


def test_capture_serializes_mm_policy_and_explicit_outputs(tmp_path):
    gm = trace(lambda x, w: ((x @ w).relu(), (x @ w).double()), (8, 16), (16, 32))
    state = _CaptureState(
        str(tmp_path),
        graph_dir=str(tmp_path / "full"),
        validate=False,
        only_include_mm=True,
    )
    state.process_graph(gm)
    state.finalize()
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["n_dropped"] == 0
    assert index["n_captured"] == 1
    assert index["captured"][0]["kind"] == "matmul"
    assert index["captured"][0]["matmul"]["outputs"] == 1
    assert index["partitioning"]["matmul_limits"]["max_reductions_chain"] == 5
    sidecar = json.loads((tmp_path / "full" / "full_graph_000.meta.json").read_text())
    assert sidecar["roundtrip"] == "ok"


def test_roundtrip_uses_serialized_mm_policy():
    gm = trace(lambda x, w: (x @ w).relu(), (8, 16), (16, 32))
    options = {"only_include_mm": True, "matmul_limits": asdict(MatmulLimits())}
    first = derive_partitions(gm, partition_options=options)
    assert len(first) == 1
    assert first == derive_partitions(gm, partition_options=options)


def test_opaque_tuple_projection_becomes_tensor_input():
    from capture_hook import compute_partition_pattern

    gm = trace(lambda x, w: (torch.linalg.qr(x).Q @ w).relu(), (16, 8), (8, 32))
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    pattern = compute_partition_pattern(region, gm)
    assert len(pattern["placeholder_info"]) == 2
    assert not any(n.op == "get_attr" for n in pattern["sub_gm"].graph.nodes)
    x, w = torch.randn(16, 8), torch.randn(8, 32)
    q = torch.linalg.qr(x).Q
    torch.testing.assert_close(pattern["sub_gm"](q, w), (q @ w).relu())


def test_horizontal_then_gating_kept_before_large_down_projection():
    def fn(x, a, b, c):
        gate = torch.nn.functional.silu(x @ a)
        up = x @ b
        return (gate * up) @ c

    gm = trace(fn, (8, 1024), (1024, 3072), (1024, 3072), (3072, 1024))
    mm = [n for n in gm.graph.nodes if is_matmul(n)]
    regions = get_fusion_partitions(gm, only_include_mm=True)
    producer = [r for r in regions if dots(r) == set(mm[:2])]
    assert producer
    assert any(n.target == torch.ops.aten.mul.Tensor for n in producer[0].outputs)
    assert all(not (set(mm) <= set(r)) for r in regions)


def test_excluded_path_cuts_before_reentry_into_region():
    def fn(x, w):
        h = x @ w
        return h.relu() + h.sin()

    gm = trace(fn, (8, 16), (16, 32))
    excluded = {n for n in gm.graph.nodes if n.target == torch.ops.aten.sin.default}
    regions = get_fusion_partitions(gm, only_include_mm=True, excluded=excluded)
    assert regions
    assert all(n.target != torch.ops.aten.add.Tensor for r in regions for n in r)
    assert regions[0].outputs[0].target == torch.ops.aten.relu.default


def test_output_pruned_repro_matches_meaningful_source_result():
    def fn(x, w):
        h = x @ w
        return h, h.relu()

    inputs = torch.randn(8, 16), torch.randn(16, 32)
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    sub, _, _ = extract_partition_subgraph(region, gm)
    torch.testing.assert_close(sub(*inputs), fn(*inputs)[1])


@pytest.mark.parametrize("extent", [128, 129])
def test_small_reduction_before_mm_is_allowed_and_large_one_is_cut(extent):
    gm = trace(
        lambda x, w: x.abs().sum(1, keepdim=True).relu() @ w, (8, extent), (1, 64)
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    targets = [n.target for n in regions[0]]
    assert torch.ops.aten.relu.default in targets
    assert (torch.ops.aten.abs.default in targets) == (extent == 128)
    assert regions[0].metrics["reduction_depth"] == (1 if extent == 128 else 0)
    assert regions[0].metrics["contraction_depth"] == (2 if extent == 128 else 1)


@pytest.mark.parametrize("extent", [128, 129])
def test_large_next_reduction_leaves_mm_pointwise_prefix(extent):
    gm = trace(lambda x, w: (x @ w).relu().sum(1).sigmoid(), (8, 1024), (1024, extent))
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    assert region.metrics["reduction_depth"] == (1 if extent == 128 else 0)
    expected = (
        torch.ops.aten.sigmoid.default if extent == 128 else torch.ops.aten.relu.default
    )
    assert region.outputs[0].target == expected


def test_mm_and_two_128_reductions_share_three_stage_budget():
    gm = trace(
        lambda x, w: (x @ w).relu().sum(1).sigmoid().sum(0).tanh(),
        (128, 1024),
        (1024, 128),
    )
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    assert region.metrics["contraction_depth"] == 3
    assert region.metrics["reduction_depth"] == 2
    assert region.metrics["max_chained_reduction_size"] == 128
    assert region.outputs[0].target == torch.ops.aten.tanh.default


def test_sixth_reduction_keeps_the_preceding_pointwise_output():
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
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    assert region.metrics["contraction_depth"] == 5
    assert region.outputs[0].target == torch.ops.aten.sigmoid.default


def test_mixed_mm_sum_mm_chain_uses_each_actual_reduction_size():
    gm = trace(
        lambda x, a, b: ((x @ a).relu().sum(1, keepdim=True).sigmoid() @ b).relu(),
        (8, 1024),
        (1024, 128),
        (1, 4096),
    )
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    assert region.metrics["mm"] == 2
    assert region.metrics["reduction_depth"] == 1
    assert region.metrics["contraction_depth"] == 3
    assert region.metrics["max_chained_reduction_size"] == 128


@pytest.mark.parametrize("second_k", [128, 129])
@pytest.mark.parametrize("third_k", [128, 129])
def test_second_and_third_mm_k_are_independently_bounded(second_k, third_k):
    gm = trace(
        lambda x, a, b, c: (((x @ a).relu() @ b).sigmoid() @ c).tanh(),
        (8, 1024),
        (1024, second_k),
        (second_k, third_k),
        (third_k, 4096),
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert any(r.metrics["mm"] == 3 for r in regions) == (second_k == third_k == 128)
    assert all(r.metrics["max_chained_reduction_size"] <= 128 for r in regions)


@pytest.mark.parametrize("kind", ["mm", "bmm", "addmm", "baddbmm"])
@pytest.mark.parametrize("extent", [128, 129])
def test_matrix_contraction_variants_follow_the_same_rule(kind, extent):
    if kind in {"bmm", "baddbmm"}:
        shapes = (2, 8, 1024), (2, 1024, extent), (2, extent, 256)

        def fn(x, a, b):
            h = torch.bmm(x, a).relu()
            return (
                torch.baddbmm(torch.zeros(2, 8, 256), h, b)
                if kind == "baddbmm"
                else torch.bmm(h, b)
            ).sigmoid()
    else:
        shapes = (8, 1024), (1024, extent), (extent, 256)

        def fn(x, a, b):
            h = (x @ a).relu()
            return (
                torch.addmm(torch.zeros(256), h, b) if kind == "addmm" else h @ b
            ).sigmoid()

    regions = get_fusion_partitions(trace(fn, *shapes), only_include_mm=True)
    assert any(r.metrics["mm"] == 2 for r in regions) == (extent == 128)


@pytest.mark.parametrize("widths", [(16, 8), (16, 16)])
def test_multiaxis_reduction_budgets_the_product_of_axes(widths):
    a, b = widths
    gm = trace(
        lambda x, w: (x @ w).view(8, a, b).relu().sum((1, 2)), (8, 1024), (1024, a * b)
    )
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    assert region.metrics["reduction_depth"] == (1 if a * b <= 128 else 0)


def test_horizontal_merge_counts_depth_per_path_not_total_mm():
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
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert regions[0].metrics["mm"] == 4
    assert regions[0].metrics["contraction_depth"] == 2


@pytest.mark.parametrize("extent", [16, 129])
def test_independent_side_reduction_uses_the_same_size_limit(extent):
    gm = trace(
        lambda x, w, y: (x @ w).relu() + y.sum(0), (8, 1024), (1024, 64), (extent, 64)
    )
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    assert region.metrics["reduction_depth"] == (1 if extent == 16 else 0)
    assert region.outputs[0].target == torch.ops.aten.add.Tensor


def test_explicit_cumulative_extent_cap_is_still_supported():
    gm = trace(lambda x, w: (x @ w).view(8, 64, 64).sum(2).sum(1), (8, 64), (64, 4096))
    regions = get_fusion_partitions(
        gm, only_include_mm=True, matmul_limits=MatmulLimits(max_reduction_extent=512)
    )
    assert regions
    assert all(r.metrics["reduction_depth"] == 1 for r in regions)


def test_tuple_reduction_prologue_lifts_each_tensor_output():
    from capture_hook import compute_partition_pattern

    def fn(x, w):
        variance, mean = torch.var_mean(x, dim=1, correction=0, keepdim=True)
        return ((x - mean) * torch.rsqrt(variance + 1e-5)) @ w

    x, w = torch.randn(8, 256), torch.randn(256, 64)
    gm = make_fx(fn, tracing_mode="fake")(x, w)
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    assert all(is_matmul(n) for n in region if is_reduction(n))
    pattern = compute_partition_pattern(region, gm)
    sub = pattern["sub_gm"]
    assert not any(n.op == "get_attr" for n in sub.graph.nodes)
    assert len(pattern["placeholder_info"]) == 4
    variance, mean = torch.var_mean(x, dim=1, correction=0, keepdim=True)
    values = {
        n.name: value
        for n, value in zip(
            [n for n in gm.graph.nodes if n.op == "placeholder"], (x, w)
        )
    }
    reduction = next(n for n in gm.graph.nodes if is_reduction(n) and not is_matmul(n))
    for n in reduction.users:
        values[n.name] = (variance, mean)[n.args[1]]
    extracted, _, _ = extract_partition_subgraph(region, gm)
    args = [values[n.name] for n in extracted.graph.nodes if n.op == "placeholder"]
    torch.testing.assert_close(sub(*args), fn(x, w))


def row_norm(x):
    centered = x - x.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(variance + 1e-5)


def test_gather_prenorm_mm_residual_postnorm_fits_one_five_reduction_chain():
    def fn(table, indices, w, residual):
        return row_norm(row_norm(table.index_select(0, indices)) @ w + residual)

    inputs = (
        torch.randn(64, 128),
        torch.arange(8),
        torch.randn(128, 128),
        torch.randn(8, 128),
    )
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    regions = get_fusion_partitions(gm, only_include_mm=True)
    full = next(r for r in regions if r.metrics["contraction_depth"] == 5)
    assert full.metrics["mm"] == 1
    assert full.metrics["reduction_depth"] == 4
    assert any(n.target == torch.ops.aten.index_select.default for n in full)
    sub, _, _ = extract_partition_subgraph(full, gm)
    torch.testing.assert_close(sub(*inputs), fn(*inputs))


def test_downstream_sum_does_not_hide_prenorm_mm_postnorm_candidate():
    def fn(table, indices, w, residual):
        y = row_norm(row_norm(table.index_select(0, indices)) @ w + residual)
        return y.sum(-1)

    gm = make_fx(fn, tracing_mode="fake")(
        torch.randn(64, 128),
        torch.arange(8),
        torch.randn(128, 128),
        torch.randn(8, 128),
    )
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    mm = next(n for n in gm.graph.nodes if is_matmul(n))
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert any(
        sum(is_reduction(n) for n in r if order[n] < order[mm]) == 2
        and sum(is_reduction(n) for n in r if order[n] > order[mm]) == 2
        and r.outputs[0].target == torch.ops.aten.mul.Tensor
        for r in regions
    )
    assert all(r.metrics["contraction_depth"] <= 5 for r in regions)


def test_horizontal_norm_mm_norm_counts_five_per_path_not_total():
    def fn(x, a, b, r1, r2):
        return row_norm(row_norm(x) @ a + r1), row_norm(row_norm(x) @ b + r2)

    gm = trace(fn, (8, 128), (128, 128), (128, 128), (8, 128), (8, 128))
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert regions[0].metrics["mm"] == 2
    assert regions[0].metrics["reduction_depth"] == 4
    assert regions[0].metrics["contraction_depth"] == 5
    assert sum(is_reduction(n) for n in regions[0]) == 10


def test_regions_without_mm_are_filtered_out():
    gm = trace(lambda x: row_norm(x).relu(), (8, 128))
    assert get_fusion_partitions(gm, only_include_mm=True) == []


def test_explicit_smaller_chain_limit_and_legacy_saved_policy():
    limits = MatmulLimits(max_reductions_chain=3)
    gm = trace(lambda x, w: row_norm(row_norm(x) @ w), (8, 128), (128, 128))
    regions = get_fusion_partitions(gm, only_include_mm=True, matmul_limits=limits)
    assert regions
    assert all(r.metrics["contraction_depth"] <= 3 for r in regions)
    assert MatmulLimits(max_contraction_depth=3).max_reductions_chain == 3
    assert "max_contraction_depth" not in asdict(MatmulLimits())


def test_first_mm_k_stays_unrestricted_with_small_weight_prenorm():
    gm = trace(lambda x, w: row_norm(x @ row_norm(w)), (8, 256), (256, 128))
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert any(r.metrics["contraction_depth"] == 5 for r in regions)


def test_large_output_fanout_keeps_a_valid_shorter_window():
    def fn(x, w):
        h = (x @ w).relu()
        return tuple(h + i for i in range(1, 7))

    gm = trace(fn, (8, 128), (128, 128))
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert regions
    assert all(len(r.outputs) <= MatmulLimits().max_outputs for r in regions)
    assert all(r.metrics["mm"] >= 1 for r in regions)


def test_nested_tuple_outputs_hash_deterministically():
    from capture_hook import pattern_hash_for_subgraph

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


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("width", [64, 32])
def test_horizontal_requires_equal_full_output_shapes_but_allows_different_k(
    batched, width
):
    prefix = (2,) if batched else ()
    gm = trace(
        lambda x, a, y, b: ((x @ a).relu(), (y @ b).relu()),
        prefix + (8, 256),
        prefix + (256, 64),
        prefix + (8, 512),
        prefix + (512, width),
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    if width == 64:
        assert len(regions) == 1
        assert regions[0].metrics["mm"] == 2
    else:
        assert len(regions) == 2
        assert all(r.metrics["mm"] == 1 for r in regions)


def test_horizontal_broadcast_join_cannot_hide_different_mm_output_shapes():
    gm = trace(
        lambda x, a, y, b: (x @ a).relu() + y @ b,
        (8, 256),
        (256, 64),
        (8, 512),
        (512, 1),
    )
    diagnostics = {}
    regions = get_fusion_partitions(gm, only_include_mm=True, diagnostics=diagnostics)
    assert regions
    assert all(r.metrics["mm"] == 1 for r in regions)
    assert diagnostics["rejections"]["parallel_output_shape"] > 0


def test_bare_horizontal_mm_with_unequal_outputs_are_dropped():
    gm = trace(lambda x, a, b: (x @ a, x @ b), (8, 256), (256, 64), (256, 32))
    assert get_fusion_partitions(gm, only_include_mm=True) == []


def test_vertical_mm_output_shapes_may_differ():
    gm = trace(
        lambda x, a, b: ((x @ a).relu() @ b).relu(), (8, 256), (256, 64), (64, 32)
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert len(regions) == 1
    assert regions[0].metrics["mm"] == 2


@pytest.mark.parametrize("epilogue", [False, True])
def test_unbind_cuts_mm_paths_and_lifts_tensor_projections(epilogue):
    from capture_hook import compute_partition_pattern

    def fn(x, a, b):
        h = (x @ a).relu()
        first, second = h.view(8, 2, 64).unbind(1)
        y = first @ b
        if epilogue:
            y = y.relu()
        return y, second

    gm = trace(fn, (8, 256), (256, 128), (64, 32))
    mm = [n for n in gm.graph.nodes if is_matmul(n)]
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert regions
    assert all(len(dots(r)) == 1 for r in regions)
    assert all(n.target != torch.ops.aten.unbind.int for r in regions for n in r)
    assert any(mm[0] in r for r in regions)
    if epilogue:
        downstream = next(r for r in regions if mm[1] in r)
        pattern = compute_partition_pattern(downstream, gm)
        placeholders = [
            n for n in pattern["sub_gm"].graph.nodes if n.op == "placeholder"
        ]
        assert len(placeholders) == 2
        assert all(isinstance(n.meta["val"], torch.Tensor) for n in placeholders)
        x, a, b = torch.randn(8, 256), torch.randn(256, 128), torch.randn(64, 32)
        first = (x @ a).relu().view(8, 2, 64).unbind(1)[0]
        torch.testing.assert_close(pattern["sub_gm"](first, b), (first @ b).relu())
    else:
        assert all(mm[1] not in r for r in regions)


def test_unbind_cannot_be_reenabled_by_the_support_callback():
    from matmul_partitioner import get_matmul_regions

    gm = trace(lambda x, w: x.unbind(0)[0].relu() @ w, (2, 8, 64), (64, 32))
    regions = get_matmul_regions(gm, lambda n: True)
    assert regions
    assert all(n.target != torch.ops.aten.unbind.int for r in regions for n in r)
    region = regions[0]
    assert len(dots(region)) == 1
    assert all(n.target is not operator.getitem for n in region)


@pytest.mark.parametrize("kind", ["padding", "bias_packing", "copy", "infinity_mask"])
def test_low_fusion_value_regions_never_reach_mm_capture(kind):
    if kind == "padding":
        gm = trace(
            lambda x, w: (x @ torch.nn.functional.pad(w, (0, 2)))[:, :-2],
            (4, 8),
            (8, 8),
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
                torch.ops.aten.select_scatter.default(
                    x.new_zeros(4, 1, 8), x, 1, 0
                ).view(4, 8)
                @ w
            ),
            (4, 8),
            (8, 8),
        )
    else:
        gm = trace(lambda x, w: torch.bmm(x, w) == -float("inf"), (2, 4, 8), (2, 8, 8))
    original = str(gm.graph)
    diagnostics = {}
    assert (
        get_fusion_partitions(gm, only_include_mm=True, diagnostics=diagnostics) == []
    )
    assert diagnostics["rejections"]["no_substantive_fusion"] > 0
    assert str(gm.graph) == original


def test_real_arithmetic_on_padded_mm_is_still_a_core_fusion():
    gm = trace(
        lambda x, w: (x @ torch.nn.functional.pad(w, (0, 2)))[:, :-2].relu(),
        (4, 8),
        (8, 8),
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert regions and any(
        n.target == torch.ops.aten.relu.default for r in regions for n in r
    )
