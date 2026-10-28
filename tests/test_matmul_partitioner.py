from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import (
    _CaptureState,
    extract_partition_subgraph,
    get_fusion_partitions,
)
from matmul_partitioner import MatmulLimits, MatmulRegion, is_matmul
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
    "shapes",
    [
        ((8, 256), (256, 64), (64, 64)),
        ((8, 64), (64, 256), (256, 64)),
        ((8, 64), (64, 64), (64, 256)),
    ],
)
def test_large_vertical_dimensions_are_cut(shapes):
    gm = trace(lambda x, a, b: ((x @ a).relu() @ b).relu(), *shapes)
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert regions
    assert all(len(dots(r)) == 1 for r in regions)


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
    "k,n,kept", [(256, 128, True), (256, 256, False), (1024, 64, False)]
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
    assert all(r.metrics["reduction_depth"] == 1 for r in regions)
    assert all(r.metrics["reduction_extent"] == 64 for r in regions)


def test_four_small_mm_do_not_multiply_reduction_extent():
    gm = trace(
        lambda x, a, b, c, d: ((((x @ a).relu() @ b).relu() @ c).relu() @ d).relu(),
        (4096, 64),
        *((64, 64),) * 4,
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert any(r.metrics["mm"] == 4 for r in regions)
    assert all(r.metrics["contraction_depth"] <= 4 for r in regions)
    assert all(r.metrics["reduction_extent"] == 1 for r in regions)


def test_mm_and_reduce_share_depth_limit():
    gm = trace(
        lambda x, a, b, c, d: ((((x @ a).relu() @ b).relu() @ c).relu() @ d).sum(1),
        (8, 64),
        *((64, 64),) * 4,
    )
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert regions
    assert all(r.metrics["contraction_depth"] <= 4 for r in regions)
    assert all(
        not (r.metrics["mm"] == 4 and r.metrics["reduction_depth"] == 1)
        for r in regions
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
    assert index["partitioning"]["matmul_limits"]["max_contraction_depth"] == 4
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


def test_excluded_path_cannot_leave_and_reenter_region():
    def fn(x, w):
        h = x @ w
        return h.relu() + h.sin()

    gm = trace(fn, (8, 16), (16, 32))
    excluded = {n for n in gm.graph.nodes if n.target == torch.ops.aten.sin.default}
    regions = get_fusion_partitions(gm, only_include_mm=True, excluded=excluded)
    assert not regions


def test_output_pruned_repro_matches_meaningful_source_result():
    def fn(x, w):
        h = x @ w
        return h, h.relu()

    inputs = torch.randn(8, 16), torch.randn(16, 32)
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    region = get_fusion_partitions(gm, only_include_mm=True)[0]
    sub, _, _ = extract_partition_subgraph(region, gm)
    torch.testing.assert_close(sub(*inputs), fn(*inputs)[1])
