"""CPU checks for explicit positive generation and external statistic domains."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from input_codec import compact_from_spec, render_T, spec_from_compact
from mm_input_constraints import annotate_external_sqrt_inputs
from repro_harness import make_inputs_from_config


def test_positive_codec_roundtrip():
    spec = {
        "kind": "tensor",
        "shape": [4, 3],
        "dtype": "float32",
        "stride": [5, 1],
        "storage_offset": 2,
        "device": "cpu",
        "gen": {"kind": "positive", "minimum": 0.25},
        "constraint": "nonnegative",
    }
    compact = compact_from_spec(spec)
    assert compact[2]["gen"] == ["positive", 0.25]
    loaded = spec_from_compact(compact)
    assert loaded["gen"] == spec["gen"]
    assert loaded["constraint"] == "nonnegative"
    assert loaded["stride"] == [5, 1] and loaded["storage_offset"] == 2
    assert "Positive(0.25)" in render_T(compact)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("stride,offset", [(None, 0), ([1, 6], 2), ([1, 0], 1)])
def test_positive_generation_preserves_layout(dtype, stride, offset):
    spec = {
        "kind": "tensor",
        "shape": [6, 4],
        "dtype": dtype,
        "device": "cpu",
        "gen": {"kind": "positive", "minimum": 0.25},
        "storage_offset": offset,
    }
    if stride is not None:
        spec["stride"] = stride
    (tensor,) = make_inputs_from_config({"inputs": [spec]})
    assert tensor.dtype == getattr(torch, dtype)
    assert tensor.shape == (6, 4)
    assert tensor.stride() == tuple(stride or [4, 1])
    assert tensor.storage_offset() == offset
    assert torch.isfinite(tensor).all() and (tensor >= 0.25).all()


def test_positive_alias_preserves_shared_storage_and_unconstrained_members():
    base = {
        "inputs": [
            {
                "kind": "tensor",
                "shape": [4, 3],
                "dtype": "float32",
                "device": "cpu",
                "stride": [6, 1],
                "storage_offset": offset,
                "alias_group": 0,
            }
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
    assert positive.storage_offset() == 0 and other.storage_offset() == 3
    torch.testing.assert_close(positive, expected, rtol=0, atol=0)
    torch.testing.assert_close(other, untouched, rtol=0, atol=0)


def test_positive_alias_broadcast_and_overlapping_constraints():
    config = {
        "inputs": [
            {
                "kind": "tensor",
                "shape": [4, 3],
                "dtype": "float32",
                "device": "cpu",
                "stride": [1, 0],
                "alias_group": 0,
                "gen": {"kind": "positive", "minimum": 0.25},
            },
            {
                "kind": "tensor",
                "shape": [4],
                "dtype": "float32",
                "device": "cpu",
                "alias_group": 0,
                "gen": {"kind": "positive", "minimum": 0.5},
            },
        ],
        "alias_group_nbytes": [4 * 4],
    }
    broadcast, flat = make_inputs_from_config(config)
    assert torch._C._is_alias_of(broadcast, flat)
    assert broadcast.stride() == (1, 0)
    assert (broadcast >= 0.5).all() and torch.isfinite(broadcast).all()
    torch.testing.assert_close(broadcast[:, 0], flat, rtol=0, atol=0)


def test_default_generation_stays_normal():
    torch.manual_seed(23)
    expected = torch.randn(32)
    torch.manual_seed(23)
    (actual,) = make_inputs_from_config(
        {
            "inputs": [
                {
                    "kind": "tensor",
                    "shape": [32],
                    "dtype": "float32",
                    "device": "cpu",
                }
            ]
        }
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual < 0).any()


def trace(fn, *args):
    gm = make_fx(fn, tracing_mode="fake")(*args)
    infos = {
        node.name: {"dtype": str(node.meta["val"].dtype)}
        for node in gm.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor)
    }
    return gm, infos


def test_external_variance_through_float_cast_view_epsilon():
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


def test_external_sqrt_and_reversed_scalar_epsilon():
    gm, infos = trace(lambda variance: torch.sqrt(1e-5 + variance), torch.ones(4))
    assert len(annotate_external_sqrt_inputs(gm, infos, minimum=0.25)) == 1
    assert next(iter(infos.values()))["gen"]["minimum"] == 0.25


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: torch.rsqrt(torch.var_mean(x, dim=-1, correction=0)[0] + 1e-5),
        lambda x: torch.rsqrt(x.square() + 1e-5),
        lambda x: torch.sqrt(x.mean(dim=-1) + 1e-5),
        lambda x: torch.rsqrt(x.int().float() + 1e-5),
        lambda x: torch.rsqrt(x - 100),
    ],
)
def test_computed_or_other_domains_are_not_rewritten(fn):
    gm, infos = trace(fn, torch.ones(3, 4))
    assert annotate_external_sqrt_inputs(gm, infos) == []
    assert all(
        "gen" not in info and "constraint" not in info for info in infos.values()
    )


def test_arbitrary_two_tensor_add_is_not_a_variance_input():
    gm, infos = trace(lambda x, y: torch.rsqrt(x + y), torch.ones(3), torch.ones(3))
    assert annotate_external_sqrt_inputs(gm, infos) == []


def test_existing_explicit_generator_is_respected():
    gm, infos = trace(lambda x: torch.rsqrt(x + 1e-5), torch.ones(3))
    info = next(iter(infos.values()))
    info["gen"] = {"kind": "constant", "value": 2.0}
    before = copy.deepcopy(infos)
    assert annotate_external_sqrt_inputs(gm, infos) == []
    assert infos == before


def test_scalar_epsilon_placeholder_remains_a_literal_domain():
    gm, infos = trace(lambda variance: torch.rsqrt(variance + 1e-5), torch.ones(4))
    add = next(node for node in gm.graph.nodes if str(node.target) == "aten.add.Tensor")
    with gm.graph.inserting_before(add):
        epsilon = gm.graph.placeholder("epsilon")
    epsilon.meta["val"] = 1e-5
    add.args = (add.args[0], epsilon)
    gm.graph.lint()
    gm.recompile()
    assert len(annotate_external_sqrt_inputs(gm, infos)) == 1


def test_generated_external_normalization_mm_inputs_are_finite():
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
    specs = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        value = node.meta["val"]
        specs.append(
            {
                "kind": "tensor",
                "shape": list(value.shape),
                "device": "cpu",
                **infos[node.name],
            }
        )
    inputs = make_inputs_from_config({"inputs": specs})
    assert torch.isfinite(gm(*inputs)).all()
    assert (inputs[0] > 0).all()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("stride", [None, [1, 5]])
def test_normal_generation_preserves_nonzero_storage_offset(dtype, stride):
    shape, offset = [5, 3], 2
    actual_stride = stride or [3, 1]
    storage_size = sum((dim - 1) * st for dim, st in zip(shape, actual_stride)) + 1
    torch.manual_seed(37)
    expected_storage = torch.randn(storage_size + offset, dtype=getattr(torch, dtype))
    expected = expected_storage.as_strided(shape, actual_stride, offset)
    spec = {
        "kind": "tensor",
        "shape": shape,
        "dtype": dtype,
        "device": "cpu",
        "storage_offset": offset,
    }
    if stride is not None:
        spec["stride"] = stride
    torch.manual_seed(37)
    (actual,) = make_inputs_from_config({"inputs": [spec]})
    assert actual.storage_offset() == offset
    assert actual.stride() == tuple(actual_stride)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_normal_packed_view_offset_survives_fake_replay():
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
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        value = placeholder.meta["val"]
        assert value.storage_offset() == 512
        assert value.stride() == tuple(spec["stride"])
        assert list(value.shape) == spec["shape"]
