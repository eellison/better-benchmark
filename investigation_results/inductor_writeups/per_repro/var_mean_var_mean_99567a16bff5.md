# var_mean_var_mean_99567a16bff5

## Summary
- **Model**: timm_swin_base_patch4_window7_224_infer
- **Pattern**: NCHW permute + dual LayerNorm (hidden=128) + Swin 7x7 window partition
- **Ratio**: 0.846x (oracle 79.49us vs compile 67.23us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle. This is the inference variant of the same Swin dual-LayerNorm + window-partition pattern as var_mean_var_mean_2989608c4b10 (training variant). The oracle's fused kernel approach is outperformed by Inductor's separate scheduling on B200 hardware. No investigation needed.
