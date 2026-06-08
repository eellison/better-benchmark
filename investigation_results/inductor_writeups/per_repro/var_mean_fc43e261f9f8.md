# var_mean_fc43e261f9f8

## Summary
- **Model**: timm_swin_base_patch4_window7_224_infer_000
- **Pattern**: Swin window-reverse + cyclic shift + residual add + LayerNorm (hidden=512)
- **Ratio**: 0.644x (oracle 44.70us vs compile 28.77us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle by a significant margin. The oracle attempted to fuse window-reverse, cyclic spatial shift (indices via iota+fmod), residual add, and full LayerNorm into one kernel. Inductor's separate scheduling of gather/layout ops and the normalization template is substantially faster on this hardware (B200). No investigation needed.
