# var_mean_d3bbb85efc8c

## Classification: BAD_ORACLE

## Current Result

- Family: `seeded_dropout_layernorm`
- Oracle path: `repros/canonical/var_mean_d3bbb85efc8c/oracle_seeded_dropout_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `37.63 us`
- `torch.compile coordinate_descent_tuning=True`: `35.74 us`
- Ratio: 0.950 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the seeded dropout LayerNorm pattern is 5% slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
