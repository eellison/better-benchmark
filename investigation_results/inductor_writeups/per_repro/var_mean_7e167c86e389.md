# var_mean_7e167c86e389

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_7e167c86e389/oracle_dropout_residual_layernorm.py`
- Correctness: PASS
- Oracle: `34.82 us`
- `torch.compile coordinate_descent_tuning=True`: `30.62 us`
- Ratio: 0.88 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the dropout residual layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
