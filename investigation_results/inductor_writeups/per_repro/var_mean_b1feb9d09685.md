# var_mean_b1feb9d09685

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_b1feb9d09685/oracle_groupnorm_affine.py`
- Correctness: PASS
- Oracle: `7.87 us`
- `torch.compile coordinate_descent_tuning=True`: `7.46 us`
- Ratio: 0.947 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the groupnorm affine pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
