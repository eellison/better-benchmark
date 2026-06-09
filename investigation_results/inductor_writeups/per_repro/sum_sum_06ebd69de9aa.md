# sum_sum_06ebd69de9aa

## Status

- Family: `multi_output_reduction`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_06ebd69de9aa/oracle_multi_output_reduction.py`
- Classification: `AT_FLOOR`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py` and returns the same three outputs:
a contiguous `float32[1000, 128]` tensor and two `float32[1000]` reduction results.

## Timings

- Oracle: 6.56 us
- torch.compile (coordinate_descent_tuning=True, combo_kernels=True): 6.72 us
- Ratio: 1.024x

## Gap Diagnosis

The ratio of 1.024x is below the 1.05x threshold. The compiled code is effectively
at the oracle floor. No performance gap exists that warrants investigation.

## Validation

- `oracle_multi_output_reduction.py --check`: PASS
- `oracle_multi_output_reduction.py --bench`: ratio 1.024x, status AT_FLOOR
