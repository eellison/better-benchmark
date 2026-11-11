import math

import pytest
import torch
import torch.nn.functional as F

from full_graph_harness import (
    FullGraphNumericsError,
    compare_outputs,
    rng_dependent_output_leaves,
    run_numerics_gate,
    validate_full_graph_numerics,
)
from numerics import (
    is_rng_call,
    judge_float_leaf,
    upcast_casts_mode,
    upcast_factory_dtypes_mode,
)
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

    _, reference = validate_full_graph_numerics(eager, eager, lambda: [torch.ones(4)])
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


def _bf16(*values):
    return torch.tensor(values, dtype=torch.bfloat16)


def _label(eager, compiled, reference):
    return judge_float_leaf(eager, compiled, reference)["label"]


def test_low_precision_compiled_within_tolerance_of_eager_or_fp32_passes():
    ref = torch.tensor([1020.0, 0.01])
    eager = _bf16(992.0, 0.01)
    assert _label(eager, eager.clone(), None) == "pass"
    # Eager lost 28 to bf16 intermediates; compiled is 24 from eager but
    # within one bf16 ulp of the fp32 reference.
    record = judge_float_leaf(eager, _bf16(1016.0, 0.01), ref)
    assert record["label"] == "pass" and record["reference"] == "fp32"


def test_low_precision_no_worse_than_eager_is_pass_precision():
    # Both are off the fp32 reference by a few ulps at different elements
    # (e.g. bf16 accumulation order); compiled's RMSE is no larger.
    torch.manual_seed(0)
    ref = torch.randn(4096) * 4
    noise = lambda: (torch.rand(4096) < 0.1) * 0.25 * torch.randn(4096).sign()
    eager = (ref + noise()).bfloat16()
    compiled = (ref + noise() * 0.9).bfloat16()
    record = judge_float_leaf(eager, compiled, ref)
    assert record["label"] == "pass_precision", record
    assert record["rmse_compiled"] <= record["rmse_eager"] * 1.05


def test_small_element_error_hidden_by_large_eager_error_fails():
    # Eager is off by 0.3 at 1000.3 (bf16 ulp 4 there); compiled is off by
    # 0.24 at the 0.01 element, below eager's max error.
    ref = torch.tensor([1000.3, 0.01] * 512)
    eager = ref.bfloat16()
    compiled = eager.clone()
    compiled[1] = 0.25
    record = judge_float_leaf(eager, compiled, ref)
    assert record["label"] == "fail", record


def test_eager_overflow_with_garbage_compiled_fails():
    ref = torch.tensor([6e4, 1.0])
    eager = torch.tensor([float("inf"), 1.0], dtype=torch.float16)
    good = torch.tensor([6e4, 1.0], dtype=torch.float16)
    assert _label(eager, good, ref) == "pass"
    for wrong in ([0.0, 5.0], [65504.0, 1.0], [6e4, float("inf")]):
        assert _label(eager, torch.tensor(wrong, dtype=torch.float16), ref) == "fail"


def test_nonfinite_where_reference_is_finite_and_nan_mismatch_fail():
    nan, inf = float("nan"), float("inf")
    ref = torch.tensor([1.0, 2.0])
    eager = _bf16(1.0, 2.0)
    assert _label(eager, _bf16(nan, 2.0), ref) == "fail"
    assert _label(eager, _bf16(inf, 2.0), ref) == "fail"
    ref_nan = torch.tensor([nan, 2.0])
    assert _label(_bf16(nan, 2.0), _bf16(nan, 2.0), ref_nan) == "pass"
    assert _label(_bf16(nan, 2.0), _bf16(0.0, 2.0), ref_nan) == "fail"


def test_value_that_rounds_to_inf_in_the_output_dtype_must_match_eager():
    inf = float("inf")
    ref = torch.tensor([1e40], dtype=torch.float64)
    assert _label(torch.tensor([inf]), torch.tensor([inf]), ref) == "pass"
    assert _label(torch.tensor([inf]), torch.tensor([3e38]), ref) == "fail"


def test_fp32_uses_eager_then_fp64_as_tiebreak():
    # Eager's reduction order lost 0.05 of the fp64 value; compiled's lost
    # less (pass_precision), or more (fail).
    ref = torch.tensor([100.0, 1.0], dtype=torch.float64)
    eager = torch.tensor([100.05, 1.0])
    assert _label(eager, eager + 1e-6, ref) == "pass"
    record = judge_float_leaf(eager, torch.tensor([100.01, 1.0]), lambda: ref)
    assert record["label"] == "pass_precision" and record["reference"] == "fp64"
    assert _label(eager, torch.tensor([100.2, 1.0]), ref) == "fail"


def test_missing_reference_fails_with_needs_reference():
    record = judge_float_leaf(torch.ones(2), torch.ones(2) * 2, lambda: None)
    assert record["label"] == "fail" and record["needs_reference"]


def test_judge_compares_imaginary_parts():
    reference = torch.tensor([1 + 2j], dtype=torch.complex128)
    eager = torch.tensor([1 + 2j])
    assert _label(eager, eager + 1j, reference) == "fail"
    assert _label(eager, eager + 1e-7j, reference) == "pass"


def test_fp32_reference_keeps_float8_quantization():
    def fake_quant(x):
        q = torch.ops.prims.convert_element_type.default(x, torch.float8_e4m3fn)
        q2 = x.to(torch.float8_e5m2)
        return q.to(torch.bfloat16), q2.to(torch.bfloat16)

    x = torch.tensor([0.3, 100.0])
    expected = [t.float() for t in fake_quant(x)]
    with upcast_casts_mode(torch.float32), upcast_factory_dtypes_mode(torch.float32):
        got = fake_quant(x)
    for g, e in zip(got, expected):
        assert g.dtype == torch.float32
        assert torch.equal(g, e)


def test_fp32_reference_leaves_float64_casts_alone():
    with upcast_casts_mode(torch.float32), upcast_factory_dtypes_mode(torch.float32):
        out = torch.ones(2, dtype=torch.bfloat16).to(torch.float64)
        buf = torch.zeros(2, dtype=torch.float16)
    assert out.dtype == torch.float64 and buf.dtype == torch.float32
