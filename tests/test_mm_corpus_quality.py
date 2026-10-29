from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from mm_corpus_quality import classify_graph_quality, classify_region_quality
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import lift_shape_params


def quality(fn, *args):
    return classify_graph_quality(make_fx(fn, tracing_mode="fake")(*args))


def test_zero_output_padding_and_exact_crop_are_alignment_controls():
    result = quality(
        lambda x, w: (x @ F.pad(w, (0, 2)))[:, :-2].float(),
        torch.randn(3, 4),
        torch.randn(4, 5),
    )
    assert result["category"] == "control"
    assert result["control_types"] == ["zero_padding"]
    assert not result["finite_input_only"]


def test_padding_remains_alignment_through_shape_preserving_cast_and_clone():
    def fn(x, w):
        padded = F.pad(w, (0, 2)).bfloat16().clone()
        return (x @ padded)[:, :-2]

    result = quality(fn, torch.randn(3, 4, dtype=torch.bfloat16), torch.randn(4, 5))
    assert result["category"] == "control"
    assert result["control_types"] == ["zero_padding"]


def test_padded_k_is_control_with_explicit_nonfinite_caveat():
    result = quality(
        lambda x, w: x @ F.pad(w, (0, 0, 0, 2)), torch.randn(3, 6), torch.randn(4, 5)
    )
    assert result["category"] == "control"
    assert result["finite_input_only"]
    assert "nonfinite" in " ".join(result["reasons"])


def test_bmm_zero_padding_then_crop_and_layout_is_an_alignment_control():
    def fn(x, attention):
        out = torch.bmm(attention, F.pad(x, (0, 0, 0, 2)))[:, :-2]
        return out.transpose(1, 2).contiguous()

    result = quality(fn, torch.randn(2, 3, 4), torch.randn(2, 5, 5))
    assert result["category"] == "control"
    assert result["control_types"] == ["zero_padding"]
    assert result["finite_input_only"]


def test_bias_packing_after_dtype_conversion_is_still_a_control():
    fn = lambda x, w, a, b: torch.addmm(torch.cat((a, b)).bfloat16(), x, w.bfloat16())
    result = quality(
        fn,
        torch.randn(3, 4, dtype=torch.bfloat16),
        torch.randn(4, 6),
        torch.randn(2),
        torch.randn(4),
    )
    assert result["category"] == "control"
    assert result["control_types"] == ["bias_packing"]


def test_nonzero_padding_negative_crop_and_uncropped_outputs_stay_core():
    x, w = torch.randn(3, 4), torch.randn(4, 5)
    for fn in (
        lambda x, w: (x @ F.pad(w, (0, 2), value=1))[:, :-2],
        lambda x, w: x @ F.pad(w, (-1, 0)),
        lambda x, w: x @ F.pad(w, (0, 2)),
        lambda x, w: (x @ F.pad(w, (0, 2)))[:, :-3],
    ):
        assert quality(fn, x, w)["category"] == "core"


def test_zero_tensor_concatenation_is_padding_but_arbitrary_weight_cat_is_core():
    x, w, tail = torch.randn(3, 4), torch.randn(4, 5), torch.randn(4, 2)
    zero = quality(
        lambda x, w: (x @ torch.cat((w, w.new_zeros(4, 2)), 1))[:, :-2], x, w
    )
    arbitrary = quality(lambda x, w, z: x @ torch.cat((w, z), 1), x, w, tail)
    assert zero["category"] == "control"
    assert arbitrary["category"] == "core"


def test_bias_packing_is_control_and_weight_packing_is_core():
    x, w, a, b = torch.randn(3, 4), torch.randn(4, 6), torch.randn(2), torch.randn(4)
    fn = lambda x, w, a, b: torch.addmm(torch.cat((a, b)), x, w)
    result = quality(fn, x, w, a, b)
    assert result["category"] == "control"
    assert result["control_types"] == ["bias_packing"]
    assert (
        quality(lambda x, w, a, b: fn(x, w, a, b).relu(), x, w, a, b)["category"]
        == "core"
    )


def test_padded_bias_and_weight_with_crop_are_controls():
    def fn(x, w, b):
        bp = torch.cat((b, b.new_zeros(2)))
        return torch.addmm(bp, x, F.pad(w, (0, 2)))[:, :-2]

    result = quality(fn, torch.randn(3, 4), torch.randn(4, 5), torch.randn(5))
    assert result["category"] == "control"
    assert result["control_types"] == ["bias_packing", "zero_padding"]


def test_full_overwrite_before_mm_is_control_but_partial_copy_is_core():
    x, w = torch.randn(3, 4), torch.randn(4, 5)

    def copy_mm(x, w):
        copied = torch.ops.aten.select_scatter.default(x.new_zeros(3, 1, 4), x, 1, 0)
        return copied.reshape(3, 4) @ w

    result = quality(copy_mm, x, w)
    assert result["category"] == "control"
    assert result["control_types"] == ["identity_input_copy"]
    assert quality(lambda x, w: copy_mm(x, w).relu(), x, w)["category"] == "core"

    def partial(x, w):
        copied = torch.ops.aten.select_scatter.default(x.new_zeros(3, 2, 4), x, 1, 0)
        return copied.reshape(6, 4) @ w

    assert quality(partial, x, w)["category"] == "core"


def test_full_slice_and_flat_strided_overwrite_are_controls():
    x, w = torch.randn(3, 4), torch.randn(4, 5)

    def slice_copy(x, w):
        return torch.ops.aten.slice_scatter.default(x.new_zeros(3, 4), x, 1, 0, 4) @ w

    def strided_copy(x, w):
        copied = torch.ops.aten.as_strided_scatter.default(
            x.new_zeros(12), x, (3, 4), (4, 1), 0
        )
        return copied.view(3, 4) @ w

    assert quality(slice_copy, x, w)["category"] == "control"
    assert quality(strided_copy, x, w)["category"] == "control"


def test_full_copy_after_mm_stays_core():
    def fn(x, w):
        y = x @ w
        return torch.ops.aten.select_scatter.default(y.new_zeros(3, 1, 5), y, 1, 0)

    assert quality(fn, torch.randn(3, 4), torch.randn(4, 5))["category"] == "core"


def test_actual_gather_activations_and_reductions_stay_core_with_padding():
    x, w, ids = torch.randn(3, 4), torch.randn(4, 5), torch.arange(2)
    assert (
        quality(
            lambda x, w, ids: (x.index_select(0, ids) @ F.pad(w, (0, 2)))[:, :-2],
            x,
            w,
            ids,
        )["category"]
        == "core"
    )
    assert (
        quality(lambda x, w: (x.relu() @ F.pad(w, (0, 2)))[:, :-2], x, w)["category"]
        == "core"
    )
    assert (
        quality(lambda x, w: (x @ F.pad(w, (0, 2)))[:, :-2].sum(-1), x, w)["category"]
        == "core"
    )
    assert (
        quality(
            lambda x, w: ((x - x.mean(-1, keepdim=True)) @ F.pad(w, (0, 2)))[:, :-2],
            x,
            w,
        )["category"]
        == "core"
    )


def test_horizontal_mm_stays_core_even_when_each_branch_is_alignment_only():
    result = quality(
        lambda x, w: ((x @ F.pad(w, (0, 2)))[:, :-2], (x @ F.pad(w, (0, 2)))[:, :-2]),
        torch.randn(3, 4),
        torch.randn(4, 5),
    )
    assert result["category"] == "core"


def test_only_bmm_infinity_equality_is_a_limited_mask_control():
    x, w = torch.randn(2, 3, 4), torch.randn(2, 4, 5)
    result = quality(lambda x, w: torch.bmm(x, w) == -math.inf, x, w)
    assert result["category"] == "control"
    assert result["finite_input_only"]
    assert "overflow" in " ".join(result["reasons"])
    assert quality(lambda x, w: torch.bmm(x, w) == 0, x, w)["category"] == "core"
    assert (
        quality(
            lambda x, w: (torch.bmm(x, w).relu(), torch.bmm(x, w) == -math.inf), x, w
        )["category"]
        == "core"
    )
    assert quality(lambda x, w: (x @ w) == -math.inf, x[0], w[0])["category"] == "core"


def test_ordinary_mm_and_single_bias_epilogue_stay_core():
    x, w, b = torch.randn(3, 4), torch.randn(4, 5), torch.randn(5)
    assert quality(lambda x, w: x @ w, x, w)["category"] == "core"
    assert quality(lambda x, w, b: x @ w + b, x, w, b)["category"] == "core"


def test_shape_lifting_preserves_padding_and_full_overwrite_classification():
    def output_padding(x, w):
        return (x @ F.pad(w, (0, 2)))[:, :-2]

    def k_padding(x, w):
        return x @ F.pad(w[:2], (0, 0, 0, 2))

    def select_copy(x, w):
        copied = torch.ops.aten.select_scatter.default(x.new_zeros(3, 1, 4), x, 1, 0)
        return copied.view(3, 4) @ w

    def strided_copy(x, w):
        copied = torch.ops.aten.as_strided_scatter.default(
            x.new_zeros(12), x, (3, 4), (4, 1), 0
        )
        return copied.view(3, 4) @ w

    def bmm_padding(x, a):
        return torch.bmm(a, F.pad(x, (0, 0, 0, 2)))[:, :-2].transpose(1, 2).contiguous()

    x, w = torch.randn(3, 4), torch.randn(4, 5)
    for fn, args in [
        (output_padding, (x, w)),
        (k_padding, (x, w)),
        (select_copy, (x, w)),
        (strided_copy, (x, w)),
        (bmm_padding, (torch.randn(2, 3, 4), torch.randn(2, 5, 5))),
    ]:
        gm = make_fx(fn, tracing_mode="fake")(*args)
        before = classify_graph_quality(gm)
        lifted, params = lift_shape_params(gm)
        assert params
        after = classify_graph_quality(lifted)
        assert before["category"] == after["category"] == "control"
        assert before["control_types"] == after["control_types"]
        assert before["finite_input_only"] == after["finite_input_only"]


def test_lifted_scalar_mask_constant_is_resolved_without_reading_tensor_values():
    gm = make_fx(lambda x, w: torch.bmm(x, w) == -math.inf, tracing_mode="fake")(
        torch.randn(2, 3, 4), torch.randn(2, 4, 5)
    )
    eq = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.eq.Scalar)
    with gm.graph.inserting_before(
        next(n for n in gm.graph.nodes if n.op == "call_function")
    ):
        scalar = gm.graph.placeholder("_scalar_param_0")
    scalar.meta["val"] = -math.inf
    eq.args = (eq.args[0], scalar)
    assert classify_graph_quality(gm)["category"] == "control"
    # A tensor-valued placeholder is not a scalar proof, even if its contents
    # happen to equal -inf. The classifier must not call item() or inspect it.
    scalar.meta["val"] = torch.tensor(-math.inf)
    assert classify_graph_quality(gm)["category"] == "core"


def test_tensor_metadata_cannot_be_used_as_a_concrete_shape_parameter():
    gm = make_fx(lambda x, w: (x @ F.pad(w, (0, 2)))[:, :-2], tracing_mode="fake")(
        torch.randn(3, 4), torch.randn(4, 5)
    )
    gm, params = lift_shape_params(gm)
    pad = next(
        n for n in gm.graph.nodes if n.target == torch.ops.aten.constant_pad_nd.default
    )
    shape_node = pad.args[1]
    assert shape_node.op == "placeholder" and params
    shape_node.meta["val"] = torch.tensor([0, 2, 0, 0])
    assert classify_graph_quality(gm)["category"] == "core"


def test_region_subsets_agree_with_standalone_controls_without_placeholders():
    def full_copy(x, w):
        copied = torch.ops.aten.as_strided_scatter.default(
            x.new_zeros(12), x, (3, 4), (4, 1), 0
        )
        return copied.view(3, 4) @ w

    x, w, b = torch.randn(3, 4), torch.randn(4, 5), torch.randn(5)
    cases = [
        (lambda x, w: (x @ F.pad(w, (0, 2)))[:, :-2], (x, w)),
        (
            lambda x, w, b: torch.addmm(
                torch.cat((b, b.new_zeros(2))), x, F.pad(w, (0, 2))
            )[:, :-2],
            (x, w, b),
        ),
        (full_copy, (x, w)),
        (
            lambda x, w: torch.bmm(x, w) == -math.inf,
            (torch.randn(2, 3, 4), torch.randn(2, 4, 5)),
        ),
    ]
    for fn, args in cases:
        gm = make_fx(fn, tracing_mode="fake")(*args)
        gm, _ = lift_shape_params(gm)
        region = {n for n in gm.graph.nodes if n.op == "call_function"}
        standalone = classify_graph_quality(gm)
        subset = classify_region_quality(region)
        assert standalone["category"] == subset["category"] == "control"
        assert standalone["control_types"] == subset["control_types"]
        assert standalone["finite_input_only"] == subset["finite_input_only"]


def test_region_treats_outside_zero_producer_as_an_opaque_input():
    def fn(x, w):
        return (x @ torch.cat((w, w.new_zeros(4, 2)), 1))[:, :-2]

    gm = make_fx(fn, tracing_mode="fake")(torch.randn(3, 4), torch.randn(4, 5))
    all_calls = {n for n in gm.graph.nodes if n.op == "call_function"}
    assert classify_region_quality(all_calls)["category"] == "control"
    without_zero_producer = {
        n for n in all_calls if n.target != torch.ops.aten.new_zeros.default
    }
    assert classify_region_quality(without_zero_producer)["category"] == "core"
    # Removing padding itself also makes its producer an ordinary MM input.
    without_cat = {
        n for n in without_zero_producer if n.target != torch.ops.aten.cat.default
    }
    assert classify_region_quality(without_cat)["category"] == "core"


def test_region_does_not_cross_external_cast_or_output_crop_paths():
    gm = make_fx(
        lambda x, w: (x @ F.pad(w, (0, 2)).float())[:, :-2], tracing_mode="fake"
    )(torch.randn(3, 4), torch.randn(4, 5, dtype=torch.bfloat16))
    calls = {n for n in gm.graph.nodes if n.op == "call_function"}
    assert classify_region_quality(calls)["category"] == "control"
    without_cast = {n for n in calls if n.target != torch.ops.aten._to_copy.default}
    assert classify_region_quality(without_cast)["category"] == "core"
    without_crop = {n for n in calls if n.target != torch.ops.aten.slice.Tensor}
    assert classify_region_quality(without_crop)["category"] == "core"


def test_region_keeps_horizontal_gather_reduction_and_pointwise_work():
    x, w, ids = torch.randn(3, 4), torch.randn(4, 5), torch.arange(2)
    cases = [
        (
            lambda x, w: (
                (x @ F.pad(w, (0, 2)))[:, :-2],
                (x @ F.pad(w, (0, 2)))[:, :-2],
            ),
            (x, w),
        ),
        (
            lambda x, w, ids: (x.index_select(0, ids) @ F.pad(w, (0, 2)))[:, :-2],
            (x, w, ids),
        ),
        (lambda x, w: (x @ F.pad(w, (0, 2)))[:, :-2].sum(-1), (x, w)),
        (lambda x, w: (x.relu() @ F.pad(w, (0, 2)))[:, :-2], (x, w)),
    ]
    for fn, args in cases:
        gm = make_fx(fn, tracing_mode="fake")(*args)
        region = {n for n in gm.graph.nodes if n.op == "call_function"}
        assert (
            classify_graph_quality(gm)["category"]
            == classify_region_quality(region)["category"]
            == "core"
        )
    assert classify_region_quality(set())["category"] == "core"
