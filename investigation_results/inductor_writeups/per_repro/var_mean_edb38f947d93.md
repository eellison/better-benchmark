# var_mean_edb38f947d93

## Classification: SWIN_SINGLETON_WINDOW_RESIDUAL_LAYERNORM

## Current Result

- Family: `swin_residual_layernorm`
- Oracle path: `repros/canonical/var_mean_edb38f947d93/oracle_swin_residual_layernorm.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle fuses the complete Swin residual add, 1024-channel population var_mean, rsqrt-based affine LayerNorm, singleton-window reshape/permute aliases, and final contiguous flatten into one row kernel. Inductor currently schedules the decomposed add/var_mean/affine/view graph through a generic normalization path with alias handling outside the fused row epilogue.

## Root cause

The repro involves a Swin Transformer stage where the window size equals the spatial size (singleton window), making the window reshape/permute operations metadata-only aliases. However, Inductor's scheduler does not canonicalize these singleton-window reshape-permute aliases early enough to sink both the residual-add producer and final flatten into the same normalization schedule. This results in the residual add being separated from the norm, requiring a re-read.

## Kernel count

- Oracle: 1 kernel (fused residual add + LayerNorm + affine + flatten store)
- Inductor: 1-2 kernels (possibly separate residual add, or suboptimal norm with re-read)

## Config exploration

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning=True | Minor tile improvement |
| multi_kernel=2 | Persistent may help for hidden=1024 |

## Recommendation

Extend the normalization scheduler to recognize Swin singleton-window aliases (where reshape/permute does not change the data layout) and emit one full-scope residual-add LayerNorm store to the flattened output. The key insight is that when windows=1x1, the permute is a no-op in terms of data movement, and the entire chain from residual-add through affine should fuse trivially.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (alias canonicalization before norm fusion)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (detect no-op reshape/permute for singleton windows)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (residual-norm kernel emission)
