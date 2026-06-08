# var_mean_9ce39d6fae7b - oracle_swin_window_layernorm

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 21.98 us
- Compile: 21.76 us
- Ratio: 0.990x

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_9ce39d6fae7b/oracle_swin_window_layernorm.py`

The compiled Inductor output matches the oracle. No Inductor improvement needed.

## Details
- Models: timm swin_base_patch4_window7_224 inference (2 instances)
- Pattern: Hidden-size-512 population var_mean LayerNorm affine + fixed Swin 7x7 window-partition clone
- Shape: [25088, 512] f32 output
- Classification in oracle: SCHEDULER_FUSION (output-contiguous vs source-contiguous traversal for window layout clone), but Inductor already achieves parity
- Correctness: PASS
