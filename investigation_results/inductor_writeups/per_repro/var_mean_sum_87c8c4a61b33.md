# var_mean_sum_87c8c4a61b33

## Summary
- **Model**: genai_LayerNormBackward_000
- **Pattern**: bf16 cast + LayerNorm (hidden=512, rows=1152000) + affine + bf16 cast + global sum
- **Ratio**: 0.721x (oracle 366.30us vs compile 263.97us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle by a wide margin (~28% faster). The oracle attempted to fuse LayerNorm with the downstream global sum reduction into a single kernel, but Inductor's separate scheduling of the normalization and whole-tensor reduction is substantially faster on this hardware (B200) for this large problem size (1152000 rows x 512 cols = 589M elements). No investigation needed.
