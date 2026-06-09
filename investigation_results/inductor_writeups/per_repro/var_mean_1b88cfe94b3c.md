# var_mean_1b88cfe94b3c

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_1b88cfe94b3c/oracle_groupnorm_residual.py`
- Correctness: PASS
- Oracle: `6.91 us`
- `torch.compile coordinate_descent_tuning=True`: `5.92 us`
- Ratio: 0.856 (oracle 17% slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's groupnorm + residual kernel for [64,256,2,2] is 17% slower than Inductor on this hardware. Inductor's generated code is more efficient for this small spatial shape.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
