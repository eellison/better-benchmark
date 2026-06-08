# var_mean_aa8ab7fa55a7

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 29.44 us
- Compile: 29.31 us
- Ratio: 0.996x

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_aa8ab7fa55a7/oracle_swin_patchmerge_layernorm.py`

The compiled Inductor output already matches the oracle performance. No Inductor improvement needed.

## Previous diagnosis (superseded by measurement)

The oracle was designed to compute Swin patch-merge residual add, fixed 2x2 layout clone, population var_mean LayerNorm, affine epilogue, and final [6272, 2048] reshape in one row kernel. Measurements show Inductor already achieves equivalent performance.

## Details
- Model: timm_swin_base_patch4_window7_224_infer
- Pattern: patch-merge layout + residual add + var_mean -> LN affine -> view
- Shape: [6272, 2048] output
- Correctness: PASS
