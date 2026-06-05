# var_mean_23c4efbb09b5

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_23c4efbb09b5/oracle_vit_patch_layernorm.py`
- Correctness: PASS
- Oracle: `19.58 us`
- `torch.compile coordinate_descent_tuning=True`: `13.76 us`
- Ratio: 0.703 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the vit patch layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
