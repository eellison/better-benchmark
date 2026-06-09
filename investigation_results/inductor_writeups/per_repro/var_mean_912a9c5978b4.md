# var_mean_912a9c5978b4

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_912a9c5978b4/oracle_seeded_dropout_layernorm.py`
- Correctness: PASS
- Oracle: `13.98 us`
- `torch.compile coordinate_descent_tuning=True`: `11.94 us`
- Ratio: 0.854 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the seeded dropout layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
