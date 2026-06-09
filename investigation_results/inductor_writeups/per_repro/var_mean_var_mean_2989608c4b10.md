# var_mean_var_mean_2989608c4b10

## Summary
- **Model**: timm_swin_base_patch4_window7_224_train
- **Pattern**: NCHW permute + dual LayerNorm (hidden=128) + Swin 7x7 window partition
- **Ratio**: 0.791x (oracle 84.90us vs compile 67.17us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle. The oracle attempted to fuse both chained LayerNorm stages (from channels-last convolution output) with the final Swin window-partition layout (permute+clone) into a single kernel. Inductor's default scheduling with separate normalization templates and layout copy outperforms this fused approach on B200 hardware. No investigation needed.
