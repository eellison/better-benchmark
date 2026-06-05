# var_mean_4f85d2df9fc7

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_4f85d2df9fc7/oracle_mobilevit_patch_layernorm.py`
- Correctness: PASS
- Oracle: `10.3 us`
- `torch.compile coordinate_descent_tuning=True`: `8.96 us`
- Ratio: 0.87 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the mobilevit patch layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
