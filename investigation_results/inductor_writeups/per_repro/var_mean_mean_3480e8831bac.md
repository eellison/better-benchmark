# var_mean_mean_3480e8831bac

## Compile: 32.9us (after fix), Oracle: 21.9us, Gap: 1.50x (was 1.83x)

## Diagnosis: PERSISTENT_REDUCTION_WARP_OVERHEAD

## Root cause: Same as var_mean_mean_2ac1c2eb8544 - the spatial mean kernel uses too many warps for a tiny reduction (rnumel=49). Inductor's norm+mean fusion is already correct (kernel 2 does NOT materialize the full normalized tensor), but the persistent reduction config picks XBLOCK=32/warps=8 when XBLOCK=16/warps=1 is significantly better.

Additionally, this repro has an extra kernel for the stochastic drop-path seed lookup (RNG). The oracle fuses the RNG inline into the normalization kernel, while Inductor generates a separate pointwise kernel for it.

## Kernel count
- Inductor: 3 kernels (RNG seed lookup, var_mean+drop_path, fused norm+mean)
- Oracle: 2 kernels (spatial-chunk norm kernel with inline RNG, finalize spatial-mean)
- Key difference: Oracle fuses RNG into kernel 1, Inductor keeps it separate

## Config exploration results
- Before fix: compile ~40.9us, ratio 1.83x
- After fix (low-warp persistent config): compile ~32.9us, ratio 1.50x
- multi_kernel=2/3: no improvement

## Fix implemented
File: `/tmp/pytorch-work/torch/_inductor/runtime/triton_heuristics.py`
Change: Add single-warp configs (XBLOCK=8,16 with num_warps=1) to `_persistent_reduction_configs()` when rnumel<=128 and xnumel>=4096.

Commit: 87bfbfce9a3 on pr-184905 (same fix as var_mean_mean_2ac1c2eb8544)

## Remaining gap (1.46x)
Two sources:
1. Same as var_mean_mean_2ac1c2eb8544: oracle uses serial-loop kernel structure vs Inductor's persistent 2D tile
2. Extra kernel: Oracle fuses RNG into the norm kernel; Inductor keeps RNG as separate pointwise kernel. This adds ~2-3us of kernel launch overhead.

## Classification: PERSISTENT_REDUCTION_WARP_OVERHEAD

Related: var_mean_mean_2ac1c2eb8544 (same core pattern without stochastic producer)

## Status: fix_implemented (partial)
