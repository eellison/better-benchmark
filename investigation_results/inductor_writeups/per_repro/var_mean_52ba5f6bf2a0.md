# var_mean_52ba5f6bf2a0

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_52ba5f6bf2a0/oracle_bn_relu_cat.py`
- Correctness: PASS
- Oracle: `14.34 us`
- `torch.compile coordinate_descent_tuning=True`: `13.15 us`
- Ratio: 0.917 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the bn relu cat pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
