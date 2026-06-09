# var_mean_8c67fcfa62ca - oracle_swin_droppath_window_layernorm

## Status: BAD_ORACLE (compile beats oracle)

- Oracle: 40.64 us
- Compile: 30.59 us
- Ratio: 0.753x

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_8c67fcfa62ca/oracle_swin_droppath_window_layernorm.py`

The compiled Inductor output outperforms the oracle by 25%. No Inductor improvement needed.

## Details
- Models: timm swin_base_patch4_window7_224 training (2 instances)
- Pattern: Swin seed-index-37 drop-path residual LayerNorm + 7x7 window-partition layout store + rsqrt/512 side output
- The oracle attempts to fuse drop-path, LayerNorm, window-partition, and side output in one kernel, but Inductor's generic scheduled approach is more efficient at this shape
- Correctness: PASS (stochastic outputs skipped in check)
