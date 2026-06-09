# var_mean_2ceed1dd2572

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_2ceed1dd2572/oracle_layernorm.py`
- Correctness: PASS
- Oracle: `6.56 us`
- `torch.compile coordinate_descent_tuning=True`: `6.11 us`
- Ratio: 0.932 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
