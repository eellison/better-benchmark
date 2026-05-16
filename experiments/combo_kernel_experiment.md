# combo_kernels=True Experiment

## Repro: pointwise_e6befa69b424 (T5 optimizer, 524 copy_ ops)

| Config | Kernels | Time (us) | vs SOL | Speedup |
|--------|---------|-----------|--------|---------|
| SOL (single memcopy) | 1 | 234 | 1.0x | — |
| Default (combo_kernels=False) | 524 | 4399 | 18.8x | 1.00x |
| combo_kernels=True | 66 | 1458 | 6.2x | **3.02x** |
| combo_kernels=True + mixed_sizes=2 | 66 | 1438 | 6.1x | **3.06x** |

## Conclusion
- `combo_kernels=True` is a near-free 3x improvement for optimizer patterns
- Config exists in PyTorch but defaults to False
- Reducing from 524 to 66 kernels eliminates most launch overhead
- Remaining gap (6.2x) from 66 launches + small irregular tensors
