"""mm_region_quality: which single-MM regions are low-value "controls"."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import lift_shape_params
from mm_region_quality import classify_graph_quality, classify_region_quality


def graph(fn, *args):
    return make_fx(fn, tracing_mode="fake")(*args)


X, W, B = torch.randn(3, 4), torch.randn(4, 5), torch.randn(5)
BMM_ARGS = torch.randn(2, 3, 4), torch.randn(2, 4, 5)


def full_strided_copy(x, w):
    copied = torch.ops.aten.as_strided_scatter.default(
        x.new_zeros(12), x, (3, 4), (4, 1), 0
    )
    return copied.view(3, 4) @ w


CONTROLS = {
    "output_padding_cropped_through_cast": (
        lambda x, w: (x @ F.pad(w, (0, 2)).bfloat16().float())[:, :-2],
        (X, W),
        ["zero_padding"],
    ),
    "k_padding": (
        lambda x, w: x @ F.pad(w, (0, 0, 0, 2)),
        (torch.randn(3, 6), W),
        ["zero_padding"],
    ),
    "zero_tensor_cat": (
        lambda x, w: (x @ torch.cat((w, w.new_zeros(4, 2)), 1))[:, :-2],
        (X, W),
        ["zero_padding"],
    ),
    "bias_packing": (
        lambda x, w, a, b: torch.addmm(torch.cat((a, b)), x, w),
        (X, torch.randn(4, 6), torch.randn(2), torch.randn(4)),
        ["bias_packing"],
    ),
    "padded_bias_and_weight": (
        lambda x, w, b: torch.addmm(
            torch.cat((b, b.new_zeros(2))), x, F.pad(w, (0, 2))
        )[:, :-2],
        (X, W, B),
        ["bias_packing", "zero_padding"],
    ),
    "full_select_copy": (
        lambda x, w: torch.ops.aten.select_scatter.default(
            x.new_zeros(3, 1, 4), x, 1, 0
        ).view(3, 4)
        @ w,
        (X, W),
        ["identity_input_copy"],
    ),
    "full_strided_copy": (full_strided_copy, (X, W), ["identity_input_copy"]),
    "bmm_infinity_mask": (
        lambda x, w: torch.bmm(x, w) == -math.inf,
        BMM_ARGS,
        ["bmm_infinity_mask"],
    ),
}

CORE = {
    "relu_prologue": (lambda x, w: (x.relu() @ F.pad(w, (0, 2)))[:, :-2], (X, W)),
    "sum_epilogue": (lambda x, w: (x @ F.pad(w, (0, 2)))[:, :-2].sum(-1), (X, W)),
    "nonzero_padding": (
        lambda x, w: (x @ F.pad(w, (0, 2), value=1))[:, :-2],
        (X, W),
    ),
    "padding_not_cropped": (lambda x, w: x @ F.pad(w, (0, 2)), (X, W)),
    "real_weight_cat": (
        lambda x, w, z: x @ torch.cat((w, z), 1),
        (X, W, torch.randn(4, 2)),
    ),
    "partial_copy": (
        lambda x, w: torch.ops.aten.select_scatter.default(
            x.new_zeros(3, 2, 4), x, 1, 0
        ).reshape(6, 4)
        @ w,
        (X, W),
    ),
    "two_mms": (
        lambda x, w: (
            (x @ F.pad(w, (0, 2)))[:, :-2],
            (x @ F.pad(w, (0, 2)))[:, :-2],
        ),
        (X, W),
    ),
    "bmm_equals_zero": (lambda x, w: torch.bmm(x, w) == 0, BMM_ARGS),
}


@pytest.mark.parametrize("name", CONTROLS)
def test_bookkeeping_only_regions_are_controls(name):
    fn, args, types = CONTROLS[name]
    assert classify_graph_quality(graph(fn, *args)) == {
        "category": "control",
        "control_types": types,
    }


@pytest.mark.parametrize("name", CORE)
def test_real_work_keeps_a_region_core(name):
    fn, args = CORE[name]
    assert classify_graph_quality(graph(fn, *args))["category"] == "core"


@pytest.mark.parametrize("name", ["output_padding_cropped_through_cast", "k_padding", "full_strided_copy"])
def test_lifted_shape_arguments_give_the_same_label(name):
    # Curation classifies graphs whose shape arguments became placeholders.
    fn, args, types = CONTROLS[name]
    lifted, params = lift_shape_params(graph(fn, *args))
    assert params
    assert classify_graph_quality(lifted)["control_types"] == types


def test_only_literal_placeholders_are_trusted_never_tensor_values():
    gm = graph(lambda x, w: torch.bmm(x, w) == -math.inf, *BMM_ARGS)
    eq = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.eq.Scalar)
    with gm.graph.inserting_before(
        next(n for n in gm.graph.nodes if n.op == "call_function")
    ):
        scalar = gm.graph.placeholder("_scalar_param_0")
    eq.args = (eq.args[0], scalar)
    scalar.meta["val"] = -math.inf
    assert classify_graph_quality(gm)["category"] == "control"
    # A tensor that happens to hold -inf is not proof; it is never read.
    scalar.meta["val"] = torch.tensor(-math.inf)
    assert classify_graph_quality(gm)["category"] == "core"


@pytest.mark.parametrize("name", ["padded_bias_and_weight", "full_strided_copy", "bmm_infinity_mask"])
def test_region_subset_matches_the_standalone_graph(name):
    fn, args, types = CONTROLS[name]
    gm, _ = lift_shape_params(graph(fn, *args))
    region = {n for n in gm.graph.nodes if n.op == "call_function"}
    assert classify_region_quality(region) == classify_graph_quality(gm)
    assert classify_region_quality(region)["control_types"] == types


def test_producers_outside_the_region_are_opaque_inputs():
    fn, args, _ = CONTROLS["zero_tensor_cat"]
    gm = graph(fn, *args)
    calls = {n for n in gm.graph.nodes if n.op == "call_function"}
    assert classify_region_quality(calls)["category"] == "control"
    # Without the zeros producer, the cat joins two unknown tensors.
    no_zeros = {n for n in calls if n.target != torch.ops.aten.new_zeros.default}
    assert classify_region_quality(no_zeros)["category"] == "core"
