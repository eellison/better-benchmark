# var_mean_5fa359332f2e

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_5fa359332f2e/oracle_dropout_residual_layernorm.py`
- Correctness: PASS
- Oracle: `12.99 us`
- `torch.compile coordinate_descent_tuning=True`: `11.9 us`
- Ratio: 0.916 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the dropout residual layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
