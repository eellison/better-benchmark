# var_mean_24e1d446bfed

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_24e1d446bfed/oracle_layernorm.py`
- Correctness: PASS
- Oracle: `6.78 us`
- `torch.compile coordinate_descent_tuning=True`: `6.24 us`
- Ratio: 0.92 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
