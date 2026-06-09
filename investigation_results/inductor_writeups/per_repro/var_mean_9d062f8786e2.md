# var_mean_9d062f8786e2

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_9d062f8786e2/oracle_residual_layernorm.py`
- Correctness: PASS
- Oracle: `10.14 us`
- `torch.compile coordinate_descent_tuning=True`: `9.5 us`
- Ratio: 0.937 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the residual layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
