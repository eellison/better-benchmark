# var_mean_d4fb0346ff2a

## Classification: AT_FLOOR

## Current Result

- Family: `m2m100_dropout_layernorm`
- Oracle path: `repros/canonical/var_mean_d4fb0346ff2a/oracle_m2m100_dropout_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `34.69 us`
- `torch.compile coordinate_descent_tuning=True`: `34.18 us`
- Ratio: 0.985
- Status: `at_floor`

## Diagnosis

Inductor's compiled code is slightly faster than the oracle for this M2M100 dropout layernorm pattern. No performance gap exists.

## Config exploration results

No further investigation needed -- compile is already faster than oracle.

## Kernel count
- Inductor at or exceeding oracle performance.

## Conclusion

Inductor is at floor. No action needed.
