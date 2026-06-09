# var_mean_7af9076c19a1

## Summary
- **Model**: ViT class token + LayerNorm
- **Pattern**: class token concat + var_mean + normalize + affine
- **Ratio**: 0.715x (oracle 25.25us vs compile 18.05us)
- **Classification**: BAD_ORACLE (oracle slower than compile)

## Result
Inductor already beats the oracle by 28.5%. No investigation needed.
