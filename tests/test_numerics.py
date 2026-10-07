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
