# var_mean_83eed019e52b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_83eed019e52b/oracle_fused_layernorm.py`
- Correctness: PASS
- Oracle: `9.09 us`
- `torch.compile coordinate_descent_tuning=True`: `7.01 us`
- Ratio: 0.771 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the fused layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
