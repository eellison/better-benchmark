"""Positive input generation and the external sqrt/rsqrt input rule."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from input_codec import compact_from_spec, render_T, spec_from_compact
from mm_input_constraints import annotate_external_sqrt_inputs
from repro_harness import make_inputs_from_config


def tensor_spec(shape, **extra):
    return {"kind": "tensor", "shape": shape, "dtype": "float32", "device": "cpu", **extra}


def trace(fn, *args):
    gm = make_fx(fn, tracing_mode="fake")(*args)
    infos = {
        node.name: {"dtype": str(node.meta["val"].dtype)}
        for node in gm.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor)
    }
    return gm, infos


# --- the "positive" generator ---


def test_positive_generator_roundtrips_through_the_codec():
    spec = tensor_spec(
        [4, 3],
        stride=[5, 1],
        storage_offset=2,
        gen={"kind": "positive", "minimum": 0.25},
        constraint="nonnegative",
    )
    compact = compact_from_spec(spec)
    assert compact[2]["gen"] == ["positive", 0.25]
    loaded = spec_from_compact(compact)
    assert loaded["gen"] == spec["gen"] and loaded["constraint"] == "nonnegative"
    assert loaded["stride"] == [5, 1] and loaded["storage_offset"] == 2
    assert "Positive(0.25)" in render_T(compact)


@pytest.mark.parametrize(
    "dtype,stride,offset",
    [("float32", None, 0), ("bfloat16", [1, 6], 2), ("float16", [1, 0], 1)],
)
def test_positive_generator_keeps_dtype_and_layout(dtype, stride, offset):
    spec = tensor_spec(
        [6, 4],
        dtype=dtype,
        gen={"kind": "positive", "minimum": 0.25},
        storage_offset=offset,
    )
    if stride is not None:
        spec["stride"] = stride
    (tensor,) = make_inputs_from_config({"inputs": [spec]})
    assert tensor.dtype == getattr(torch, dtype)
    assert tensor.stride() == tuple(stride or [4, 1])
    assert tensor.storage_offset() == offset
    assert torch.isfinite(tensor).all() and (tensor >= 0.25).all()


def test_positive_alias_member_leaves_the_other_member_untouched():
    base = {
        "inputs": [
            tensor_spec([4, 3], stride=[6, 1], storage_offset=offset, alias_group=0)
            for offset in [0, 3]
        ],
        "alias_group_nbytes": [4 * 6 * 4],
    }
    torch.manual_seed(17)
    expected, untouched = make_inputs_from_config(base)
    expected, untouched = expected.abs() + 0.25, untouched.clone()
    constrained = copy.deepcopy(base)
    constrained["inputs"][0]["gen"] = {"kind": "positive", "minimum": 0.25}
    torch.manual_seed(17)
    positive, other = make_inputs_from_config(constrained)
    assert torch._C._is_alias_of(positive, other)
    torch.testing.assert_close(positive, expected, rtol=0, atol=0)
    torch.testing.assert_close(other, untouched, rtol=0, atol=0)


def test_overlapping_positive_aliases_use_the_larger_minimum():
    config = {
        "inputs": [
            tensor_spec(
                [4, 3],
                stride=[1, 0],
                alias_group=0,
                gen={"kind": "positive", "minimum": 0.25},
            ),
            tensor_spec([4], alias_group=0, gen={"kind": "positive", "minimum": 0.5}),
        ],
        "alias_group_nbytes": [4 * 4],
    }
    broadcast, flat = make_inputs_from_config(config)
    assert torch._C._is_alias_of(broadcast, flat)
    assert broadcast.stride() == (1, 0)
    assert (broadcast >= 0.5).all() and torch.isfinite(broadcast).all()
    torch.testing.assert_close(broadcast[:, 0], flat, rtol=0, atol=0)


def test_default_generation_is_unchanged():
    torch.manual_seed(23)
    expected = torch.randn(32)
    torch.manual_seed(23)
    (actual,) = make_inputs_from_config({"inputs": [tensor_spec([32])]})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,stride", [("float32", None), ("bfloat16", [1, 5])])
def test_normal_generation_keeps_the_storage_offset(dtype, stride):
    shape, offset = [5, 3], 2
    actual_stride = stride or [3, 1]
    storage_size = sum((dim - 1) * st for dim, st in zip(shape, actual_stride)) + 1
    torch.manual_seed(37)
    storage = torch.randn(storage_size + offset, dtype=getattr(torch, dtype))
    expected = storage.as_strided(shape, actual_stride, offset)
    spec = tensor_spec(shape, dtype=dtype, storage_offset=offset)
    if stride is not None:
        spec["stride"] = stride
    torch.manual_seed(37)
    (actual,) = make_inputs_from_config({"inputs": [spec]})
    assert actual.storage_offset() == offset
    assert actual.stride() == tuple(actual_stride)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_storage_offset_survives_fake_tensor_replay():
    from roundtrip_validation import fake_tensor_ctx

    spec = {
        "kind": "tensor",
        "shape": [2048, 8, 49, 32],
        "dtype": "bfloat16",
        "device": "cuda",
        "stride": [37632, 32, 768, 1],
        "storage_offset": 512,
    }
    with fake_tensor_ctx():
        (tensor,) = make_inputs_from_config({"inputs": [spec]})
        gm = make_fx(lambda x: x.neg(), tracing_mode="fake")(tensor)
        value = next(n for n in gm.graph.nodes if n.op == "placeholder").meta["val"]
        assert value.storage_offset() == 512
        assert value.stride() == tuple(spec["stride"])


# --- which inputs get the positive generator ---


def test_external_variance_behind_cast_view_and_epsilon_is_annotated():
    def fn(variance, x, weight):
        stat = variance.float().view(-1, 1).expand(3, 4)
        return (x * torch.rsqrt(stat + 1e-5)) @ weight

    gm, infos = trace(
        fn, torch.ones(3, dtype=torch.bfloat16), torch.randn(3, 4), torch.randn(4, 2)
    )
    names = annotate_external_sqrt_inputs(gm, infos)
    assert names == [next(n.name for n in gm.graph.nodes if n.op == "placeholder")]
    assert infos[names[0]] == {
        "dtype": "torch.bfloat16",
        "constraint": "nonnegative",
        "gen": {"kind": "positive", "minimum": 0.1},
    }
    assert sum("gen" in info for info in infos.values()) == 1


def test_lifted_epsilon_placeholder_counts_as_a_literal():
    gm, infos = trace(lambda variance: torch.rsqrt(variance + 1e-5), torch.ones(4))
    add = next(node for node in gm.graph.nodes if str(node.target) == "aten.add.Tensor")
    with gm.graph.inserting_before(add):
        epsilon = gm.graph.placeholder("epsilon")
    epsilon.meta["val"] = 1e-5
    add.args = (add.args[0], epsilon)
    gm.graph.lint()
    gm.recompile()
    assert len(annotate_external_sqrt_inputs(gm, infos)) == 1


@pytest.mark.parametrize(
    "fn,arity",
    [
        (lambda x: torch.rsqrt(torch.var_mean(x, dim=-1, correction=0)[0] + 1e-5), 1),
        (lambda x: torch.rsqrt(x.square() + 1e-5), 1),
        (lambda x: torch.rsqrt(x.int().float() + 1e-5), 1),
        (lambda x: torch.rsqrt(x - 100), 1),
        (lambda x, y: torch.rsqrt(x + y), 2),
    ],
)
def test_computed_or_arbitrary_values_are_left_alone(fn, arity):
    gm, infos = trace(fn, *[torch.ones(3, 4)] * arity)
    assert annotate_external_sqrt_inputs(gm, infos) == []
    assert all("gen" not in info and "constraint" not in info for info in infos.values())


def test_existing_explicit_generator_is_respected():
    gm, infos = trace(lambda x: torch.rsqrt(x + 1e-5), torch.ones(3))
    next(iter(infos.values()))["gen"] = {"kind": "constant", "value": 2.0}
    before = copy.deepcopy(infos)
    assert annotate_external_sqrt_inputs(gm, infos) == []
    assert infos == before


def test_annotated_normalization_mm_inputs_give_finite_results():
    def fn(variance, x, mean, gamma, beta, weight):
        normalized = (x - mean) * torch.rsqrt(variance + 1e-5)
        return ((normalized * gamma + beta).to(torch.bfloat16) @ weight).float()

    gm, infos = trace(
        fn,
        torch.ones(8, 1),
        torch.randn(8, 16),
        torch.randn(8, 1),
        torch.randn(16),
        torch.randn(16),
        torch.randn(16, 8, dtype=torch.bfloat16),
    )
    assert len(annotate_external_sqrt_inputs(gm, infos)) == 1
    specs = [
        {
            "kind": "tensor",
            "shape": list(node.meta["val"].shape),
            "device": "cpu",
            **infos[node.name],
        }
        for node in gm.graph.nodes
        if node.op == "placeholder"
    ]
    inputs = make_inputs_from_config({"inputs": specs})
    assert torch.isfinite(gm(*inputs)).all()
    assert (inputs[0] > 0).all()
