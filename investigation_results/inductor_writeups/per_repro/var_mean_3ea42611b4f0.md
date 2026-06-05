# var_mean_3ea42611b4f0

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_3ea42611b4f0/oracle_reflect_instance_norm.py`
- Correctness: PASS
- Oracle: `10.98 us`
- `torch.compile coordinate_descent_tuning=True`: `10.27 us`
- Ratio: 0.936 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the reflect instance norm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
