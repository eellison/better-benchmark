# var_mean_e4d2655a5279

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_e4d2655a5279/oracle_channel_layernorm.py`
- Correctness: PASS
- Oracle: `48.7 us`
- `torch.compile coordinate_descent_tuning=True`: `20.1 us`
- Ratio: 0.413 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the channel layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
