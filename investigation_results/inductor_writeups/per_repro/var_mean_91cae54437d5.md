# var_mean_91cae54437d5

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_91cae54437d5/oracle_residual_layernorm.py`
- Correctness: PASS
- Oracle: `9.79 us`
- `torch.compile coordinate_descent_tuning=True`: `8.1 us`
- Ratio: 0.827 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the residual layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
