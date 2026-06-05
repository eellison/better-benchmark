# var_mean_b7b154e2f8c6

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_b7b154e2f8c6/oracle_residual_layernorm_aliases.py`
- Correctness: PASS
- Oracle: `10.85 us`
- `torch.compile coordinate_descent_tuning=True`: `9.63 us`
- Ratio: 0.888 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the residual layernorm aliases pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
