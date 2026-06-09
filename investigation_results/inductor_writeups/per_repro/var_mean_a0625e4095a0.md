# var_mean_a0625e4095a0

## Status: BAD_ORACLE (compile already faster)

- Oracle: 255.58 us
- Compile: 163.42 us
- Ratio: 0.639x (oracle is 1.56x slower than compile)

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_a0625e4095a0/oracle_swin_shifted_window_residual_layernorm.py`

The compiled Inductor output already significantly outperforms this oracle. The oracle's monolithic fused kernel (combining residual add, LayerNorm, cyclic shift, window partition, clone, and side outputs in one kernel) suffers from register pressure or suboptimal occupancy at this large shape (401408 rows x 128 hidden), while Inductor's decomposed approach is more efficient.

## Previous diagnosis (superseded by measurement)

The oracle was designed to compute the complete Swin residual add, hidden-size-128 population LayerNorm, affine epilogue, cyclic +3 spatial shift, 7x7 window partition clone, final contiguous flatten, and sibling rsqrt/128 output in one row-reduction Triton kernel. However, at this specific shape the compile output already wins.

## Details
- Model: timm_swin (training)
- Pattern: residual add -> var_mean -> LN affine -> cyclic shift -> window partition -> view
- Shape: [401408, 128] output, [128, 3136, 128] strided input
- Correctness: PASS
