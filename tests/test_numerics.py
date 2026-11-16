import math

import pytest
import torch
import torch.nn.functional as F

from full_graph_harness import (
    FullGraphNumericsError,
    _fp64_reference,
    compare_outputs,
    rng_dependent_output_leaves,
    run_numerics_gate,
    validate_full_graph_numerics,
)
from numerics import is_rng_call
from oracle_harness import _anchored_numerics_gate


def test_repeated_input_tensor_reaches_eager_and_compiled_as_one_tensor():
    seen = []

    def eager(a, b):
        seen.append(a is b)
        return a + b

    validate_full_graph_numerics(eager, eager, lambda: (lambda x: [x, x])(torch.ones(4)))
    assert seen and all(seen)


def test_conjugate_output_mismatch_is_caught():
    def eager(x):
        return x.conj()

    def compiled(x):
        return x

    with pytest.raises(FullGraphNumericsError):
        validate_full_graph_numerics(eager, compiled, lambda: [torch.tensor([1 + 2j])])


def test_compiled_may_be_finite_where_eager_overflows_and_fp64_is_finite():
    reference = torch.tensor([7e4, 1.0], dtype=torch.float64)
    eager = torch.tensor([float("inf"), 1.0], dtype=torch.float16)
    compiled = torch.tensor([65504.0, 1.0], dtype=torch.float16)
    ok, details = compare_outputs(reference, eager, compiled)
    assert ok, details


def test_float8_outputs_are_compared():
    def eager(x):
        return (x * 2).to(torch.float8_e4m3fn)

    def wrong(x):
        return (x * 3).to(torch.float8_e4m3fn)

    validate_full_graph_numerics(eager, eager, lambda: [torch.ones(4)])
    with pytest.raises(FullGraphNumericsError):
        validate_full_graph_numerics(eager, wrong, lambda: [torch.ones(4)])


def test_bitcast_output_falls_back_to_eager_reference():
    def eager(x):
        return x.view(torch.int16).view(torch.bfloat16)

    _, reference, _, _ = validate_full_graph_numerics(eager, eager, lambda: [torch.ones(4)])
    assert reference == "eager"


def test_complex_outputs_use_a_tolerance():
    def eager(x):
        return x * 3

    def close(x):
        return x * (3 + 1e-7)

    validate_full_graph_numerics(eager, close, lambda: [torch.tensor([1 + 2j])])


@pytest.mark.parametrize(
    "draw",
    [
        lambda t: t.random_(0, 100),
        lambda t: t.cauchy_(),
        lambda t: t.geometric_(0.5),
        lambda t: t.log_normal_(),
        lambda t: t.exponential_(),
    ],
)
def test_rng_ops_outside_the_old_list_are_skipped(draw):
    def eager(value):
        return value + draw(torch.empty_like(value)), value * 2

    def compiled(value):
        torch.rand(1)  # a different RNG stream than eager
        return eager(value)

    summary, _ = run_numerics_gate(
        eager, compiled, lambda: [torch.zeros(16)], label="default compiled",
        variant="default", seeds=1,
    )
    assert summary["status"] == "numerics_unavailable"
    assert summary["skipped_rng_output_leaves"] == [0]


def test_attention_without_dropout_is_not_rng():
    def attention(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    q = torch.randn(1, 2, 8, 16)
    assert rng_dependent_output_leaves(attention, lambda: [q, q.clone(), q.clone()]) == set()


def test_dropout_rng_depends_on_training_and_probability():
    dropout = torch.ops.aten.native_dropout.default
    x = torch.ones(2)
    assert is_rng_call(dropout, (x, 0.5, True))
    assert not is_rng_call(dropout, (x, 0.5, False))
    assert not is_rng_call(dropout, (x, 0.0, True))
    assert is_rng_call(torch.ops.aten.random_.default, (x,))


def test_oracle_gate_records_are_unchanged():
    f64 = torch.float64
    oracle = [torch.tensor([1.0, 3.0]), torch.tensor([5.5]), torch.tensor([1, 2]),
              torch.tensor([1.25]), torch.tensor([float("nan")]), torch.tensor([9.0])]
    compiled = [torch.tensor([1.0, 2.25]), torch.tensor([3.0]), torch.tensor([1, 3]),
                torch.tensor([0.25]), torch.tensor([4.0]), torch.tensor([0.0])]
    ref = [torch.tensor([1.0, 2.0], dtype=f64), torch.tensor([3.5], dtype=f64),
           torch.tensor([1, 2]), torch.tensor([0.25], dtype=f64),
           torch.tensor([4.0], dtype=f64), torch.tensor([0.0], dtype=f64)]

    result = _anchored_numerics_gate(oracle, compiled, ref, {5})

    nan_entry = result["per_output"][4]
    assert math.isnan(nan_entry.pop("err_oracle"))
    assert result == {
        "pass": False,
        "per_output": [
            {"idx": 0, "err_oracle": 1.0, "err_compiled": 0.25, "threshold": 0.75,
             "ref_absmax": 2.0, "pass": False},
            {"idx": 1, "err_oracle": 2.0, "err_compiled": 0.5, "threshold": 1.5,
             "ref_absmax": 3.5, "pass": False},
            {"idx": 2, "skip": "non_float"},
            {"idx": 3, "err_oracle": 1.0, "err_compiled": 0.0, "threshold": 1e-05,
             "ref_absmax": 0.25, "pass": False},
            {"idx": 4, "err_compiled": 0.0, "threshold": 4e-05, "ref_absmax": 4.0,
             "pass": False},
            {"idx": 5, "skip": "stochastic"},
        ],
        "worst_output_idx": 3,
        "ref_precision": "f64",
    }


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_math_matches_sdpa_and_its_gradients(is_causal):
    from numerics import _flash_attention_backward_math, _flash_attention_math

    q, k, v, grad = (torch.randn(2, 3, 5, 4, dtype=torch.float64, requires_grad=True) for _ in range(4))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    out, lse, *_ = _flash_attention_math(q, k, v, 0.0, is_causal)
    torch.testing.assert_close(out, ref)
    expected = torch.autograd.grad(ref, (q, k, v), grad)
    got = _flash_attention_backward_math(grad, q, k, v, out, lse, None, None, 5, 5, 0.0, is_causal, None, None)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b)


def test_reference_runs_flash_attention_as_its_math(monkeypatch):
    import numerics

    # mm stands in for flash attention (CPU has no flash kernel): the mode runs the math at the reference dtype
    seen = []
    monkeypatch.setitem(numerics._FLASH_ATTENTION_MATH, torch.ops.aten.mm.default,
                        lambda a, b: seen.append(a.dtype) or (a.unsqueeze(-1) * b).sum(1))
    monkeypatch.setattr(numerics, "_flash_math_applies", lambda *args: True)
    a, b = torch.randn(3, 4, dtype=torch.bfloat16), torch.randn(4, 5, dtype=torch.bfloat16)
    reference, error = _fp64_reference(lambda x, y: torch.mm(x, y), lambda: (a, b), 0, torch.float32)
    assert error is None and seen == [torch.float32]
    torch.testing.assert_close(reference, a.float() @ b.float())


def test_rng_state_outputs_are_reported_without_tainting_consumers(monkeypatch):
    import numerics

    # var_mean stands in for an attention op whose output 1 is its RNG state (CPU has no flash attention)
    monkeypatch.setitem(numerics._RNG_STATE_OUTPUTS, "aten::var_mean", (1,))

    def graph(x):
        var, state = torch.ops.aten.var_mean.correction(x, [0])
        return var + 1, state, x * state

    rng_state: set[int] = set()
    assert numerics.scan_rng_dependence(graph, [torch.randn(4, 2)], rng_state) == {1}
    assert rng_state == {1}


def test_unwritten_empty_outputs_are_skipped_but_not_justified():
    import numerics

    def graph(x):
        written = torch.ops.aten.empty.memory_format([4], dtype=torch.float32)
        written[:2].copy_(x[:2])
        return x + 1, torch.ops.aten.empty.memory_format([4, 1], dtype=torch.float32), written

    rng_state: set[int] = set()
    assert numerics.scan_rng_dependence(graph, [torch.randn(4)], rng_state) == {1}
    assert rng_state == set()


def test_float64_reference_runs_custom_ops_without_float64_kernels_in_float32():
    lib = torch.library.Library("bb_test_fp64", "FRAGMENT")
    lib.define("scale2(Tensor x) -> Tensor")

    def scale2(x):
        if x.dtype == torch.float64:
            raise NotImplementedError("scale2 not implemented for 'Double'")
        return x * 2

    lib.impl("scale2", scale2, "CompositeExplicitAutograd")

    def graph(x):
        return torch.ops.bb_test_fp64.scale2.default(x) + 1

    _, reference, _, _ = validate_full_graph_numerics(graph, graph, lambda: [torch.randn(4)])
    assert reference == "fp64"


def test_reference_keeps_integer_bitcasts_and_refuses_float_ones():
    def graph(x, packed):
        return x * packed.view(torch.bfloat16).float().abs().mean()

    def make_inputs():
        return torch.randn(64), torch.randint(0, 2**14, (64,), dtype=torch.int16)

    reference, error = _fp64_reference(graph, make_inputs, 0)
    assert error is None
    torch.manual_seed(0)
    torch.testing.assert_close(reference, graph(*make_inputs()).double(), rtol=1e-2, atol=0)
    with pytest.raises(FullGraphNumericsError):
        validate_full_graph_numerics(graph, lambda x, p: graph(x, p) * 2, make_inputs)

    reference, error = _fp64_reference(lambda x: x.view(torch.int32), lambda: (torch.randn(4),), 0)
    assert reference is None and "bitcast" in error


def test_reference_upcasts_integer_to_float_copies():
    def graph(idx, w):
        return idx.to(torch.bfloat16) @ w

    reference, error = _fp64_reference(
        graph, lambda: (torch.randint(0, 4, (2, 3)), torch.randn(3, 2).bfloat16()), 0)
    assert error is None and reference.dtype == torch.float64


def test_reference_does_not_retry_custom_ops_that_write_their_arguments_in_float32():
    lib = torch.library.Library("bb_test_fp64_mut", "FRAGMENT")
    lib.define("fill2(Tensor(a!) out, Tensor x) -> ()")

    def fill2(out, x):
        if x.dtype == torch.float64:
            raise NotImplementedError("fill2 not implemented for 'Double'")
        out.copy_(x * 2)

    lib.impl("fill2", fill2, "CompositeExplicitAutograd")

    def graph(x):
        out = torch.zeros_like(x)
        torch.ops.bb_test_fp64_mut.fill2.default(out, x)
        return out + 1

    reference, error = _fp64_reference(graph, lambda: (torch.randn(4),), 0)
    assert reference is None and "NotImplementedError" in error


def test_empty_zero_element_outputs_are_not_unwritten():
    import numerics

    assert numerics.scan_rng_dependence(lambda x: (x + 1, x.new_empty([0])), [torch.randn(4)]) == set()


@pytest.mark.parametrize("args, kwargs, applies", [
    ((0.0, False), {}, True),
    ((0.1, False), {}, False),
    ((0.0, True), {}, True),
    ((0.0, True), {"key_len": 6}, False),
])
def test_flash_attention_math_only_without_dropout_and_with_aligned_causal(args, kwargs, applies):
    import numerics

    q = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    k = torch.randn(1, 2, kwargs.get("key_len", 5), 4, dtype=torch.float64)
    forward = torch.ops.aten._scaled_dot_product_flash_attention.default
    backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default
    assert numerics._flash_math_applies(forward, (q, k, k, *args), {}) is applies
    dropout_p, is_causal = args
    bwd_args = (q, q, k, k, q, q, None, None, 5, 5, dropout_p, is_causal, None, None)
    assert numerics._flash_math_applies(backward, bwd_args, {}) is applies


def test_float32_reference_keeps_float64_values():
    def graph(t, x):
        return ((t - 1.7e9).float() * x).to(torch.bfloat16), t.sum(dtype=torch.float64)

    t = torch.full((4,), 1.7e9, dtype=torch.float64) + torch.arange(4, dtype=torch.float64) / 8
    reference, error = _fp64_reference(graph, lambda: (t, torch.ones(4, dtype=torch.bfloat16)), 0, torch.float32)
    assert error is None
    assert reference[0].dtype == torch.float32 and reference[0].tolist() == [0.0, 0.125, 0.25, 0.375]
    assert reference[1].dtype == torch.float64

