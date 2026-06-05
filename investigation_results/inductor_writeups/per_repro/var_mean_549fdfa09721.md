# var_mean_549fdfa09721

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_549fdfa09721/oracle_dropout_residual_layernorm_saved_scale.py`
- Correctness: PASS
- Oracle: `21.47 us`
- `torch.compile coordinate_descent_tuning=True`: `17.31 us`
- Ratio: 0.806 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the dropout residual layernorm saved scale pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
