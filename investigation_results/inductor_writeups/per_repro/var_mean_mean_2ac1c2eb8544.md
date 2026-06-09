# var_mean_mean_2ac1c2eb8544

## Compile: 27.5us (after fix), Oracle: 24.3us, Gap: 1.13x (was 1.54x)

## Diagnosis: PERSISTENT_REDUCTION_WARP_OVERHEAD

## Root cause: Inductor already correctly fuses the normalization + spatial mean into one kernel (kernel 1) without materializing the [128,49,1024] normalized intermediate. Both Inductor and the oracle use 2 kernels with the same structure. The gap is purely in the **tiling/config** of the second kernel: Inductor's persistent reduction heuristic assigns too many warps (8) to a kernel with a tiny reduction dimension (rnumel=49, xnumel=131072). The oracle uses an equivalent structure but with 1 warp and a wider feature tile, avoiding shared-memory synchronization overhead.

Specifically, the default `num_warps = total_numel() // 128` formula gives warps=8 for XBLOCK=32 with R0_BLOCK=64. But with rnumel=49, each program does very little reduction work, and multi-warp overhead dominates. XBLOCK=8-16 with num_warps=1 is 30-40% faster because single-warp programs avoid shared memory synchronization and allow more blocks to run concurrently per SM.

## Kernel count
- Inductor: 2 kernels (var_mean stats, then fused norm+spatial_mean)
- Oracle: 2 kernels (row stats, then spatial-mean-with-inline-normalization)
- Note: Inductor's kernel 1 does NOT materialize the normalized tensor; it re-reads inputs + uses pre-computed stats inline.

## Config exploration results
- Before fix: XBLOCK=32 warps=8 (compile: ~37.7us, ratio 1.54x)
- After fix: XBLOCK=16 warps=1 (compile: ~27.5us, ratio 1.13x)
- multi_kernel=2/3: no improvement
- max_autotune: no improvement (same configs explored)

## Fix implemented
File: `/tmp/pytorch-work/torch/_inductor/runtime/triton_heuristics.py`
Change: Add single-warp configs (XBLOCK=8,16 with num_warps=1) to `_persistent_reduction_configs()` when rnumel<=128 and xnumel>=4096.

Commit: 87bfbfce9a3 on pr-184905

## Remaining gap (1.25x)
The oracle uses a fundamentally different kernel structure for kernel 2:
- 2D grid (batch, feature_chunks) with a serial `for token in tl.static_range(0, 49)` loop
- 1D accumulator [BLOCK_FEATURE=64] instead of 2D tile [XBLOCK, R0_BLOCK]
- Only 2048 programs vs Inductor's 8192

This remaining gap requires a codegen enhancement to emit "serial-loop reduction" kernels instead of the standard persistent 2D tile for tiny reduction dimensions with large x.

## Classification: PERSISTENT_REDUCTION_WARP_OVERHEAD

Related: var_mean_mean_3480e8831bac (same pattern with stochastic producer, 1.83x->1.46x)

## Status: fix_implemented (partial)
