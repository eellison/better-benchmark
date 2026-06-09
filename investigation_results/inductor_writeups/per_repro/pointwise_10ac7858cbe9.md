# pointwise_10ac7858cbe9 — SwiGLU pointwise

## Summary
- **Repro**: pointwise_10ac7858cbe9
- **Ratio**: 0.99 (oracle 9.15us vs compile 9.06us)
- **Status**: AT_FLOOR

## Result
Inductor matches or slightly beats the oracle. No performance gap exists.
The compiled SwiGLU pointwise kernel is already optimal.
