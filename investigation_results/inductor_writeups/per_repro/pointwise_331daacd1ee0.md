# pointwise_331daacd1ee0

## Classification: BAD_ORACLE

## Current Result

- Family: `dropout_residual`
- Oracle path: `repros/canonical/pointwise_331daacd1ee0/oracle_dropout_residual.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `24.06 us`
- `torch.compile coordinate_descent_tuning=True`: `22.24 us`
- Ratio: 0.924 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the dropout residual pattern on shape [16, 128, 2560] is 8% slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
