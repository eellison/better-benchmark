# var_mean_b95d34ef7580

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_b95d34ef7580/oracle_mobilevit_reverse_layernorm.py`
- Correctness: PASS
- Oracle: `17.31 us`
- `torch.compile coordinate_descent_tuning=True`: `16.29 us`
- Ratio: 0.941 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the mobilevit reverse layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
