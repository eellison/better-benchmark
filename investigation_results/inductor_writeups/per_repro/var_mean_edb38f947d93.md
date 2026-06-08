# var_mean_edb38f947d93

## Classification: AT_FLOOR

## Current Result

- Family: `swin_residual_layernorm`
- Oracle path: `repros/canonical/var_mean_edb38f947d93/oracle_swin_residual_layernorm.py`
- Correctness: PASS (max_diff=2.86e-06)
- Oracle: `17.92 us`
- `torch.compile coordinate_descent_tuning=True`: `17.86 us`
- Ratio: 0.996 (compile matches oracle within noise)
- Status: `at_floor`

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.0, compile is actually marginally faster). The Swin residual LayerNorm pattern at [6272, 1024] shape is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled. No performance gap exists.

The singleton-window reshape/permute operations in this pattern are correctly identified as metadata-only aliases, and Inductor fuses the residual add + LayerNorm + affine into one efficient kernel.

## Config exploration results

- Baseline compile already at oracle level; no further config exploration needed.

## Relevant files

- Oracle: `repros/canonical/var_mean_edb38f947d93/oracle_swin_residual_layernorm.py`
