# sum_sum_sum_6b7ff2251678


## Measured Timings
- Oracle: 7.74 us
- Compile (CDT): 8.83 us
- Ratio: 1.14x

## Summary

- Model: ConvNeXt layernorm-backward (multi-output)
- Oracle: `oracle_multi_output.py`
- Classification: MULTI_OUTPUT_SHARED_REDUCTION
- Ratio: 1.694x -> **1.17x** (with fix)
- Kernel count: Inductor 3 kernels -> 2 kernels (Oracle: 1 kernel)

## Root Cause

The oracle computes the full ConvNeXt layernorm-backward-shaped graph in one row-local multi-output Triton reduction, including:
1. Reshape/permute pointwise work
2. Two per-row reductions
3. Two sibling channel reductions
4. The as_strided_scatter/expand/div path feeding the third channel reduction

Inductor (before fix) emits 3 separate kernels:
1. Combo kernel: row-local reductions (sum over channel dim) + first two channel reductions (sum over batch)
2. Spatial expansion kernel: expand [128,640,1,1]->[128,640,7,7], div by 49, partial sum over batch*spatial
3. Finalization kernel: sum partials for the third channel output

The third output's computation `sum(expand(x, [128,640,7,7]) / 49, [0,2,3])` is algebraically
equivalent to `sum(x, [0])` because the expand+sum over spatial dims multiplies by 49 and
the division by 49 cancels. After the fix, this simplification is recognized and the third
output is computed as a simple channel reduction alongside the other two.

## Fix Implemented

**Expand-sum elision FX pass** (`torch/_inductor/fx_passes/expand_sum_elision.py`)

The pass recognizes the pattern:
```
sum(div(expand(x, shape), N), dims)
```
where the expanded dims (input size 1, output size > 1) are all included in the sum dims,
and N equals the product of those expanded sizes. Rewrites to:
```
sum(x, remaining_dims)
```

This eliminates the expensive spatial expansion kernel entirely. The third channel reduction
is now fused into the second kernel alongside the other channel reductions.

### Commit: `c33b0e78618` in `/tmp/pytorch-work`
- `torch/_inductor/fx_passes/expand_sum_elision.py` (new file)
- `torch/_inductor/fx_passes/post_grad.py` (register the pass)
- `torch/_inductor/config.py` (add `expand_sum_elision` flag, default True)

## Config Exploration (after fix)

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 8.99 | 1.17x |
| multi_kernel=2 | ~33 | ~4.3x (WORSE) |
| multi_kernel=3 | ~32 | ~4.1x (WORSE) |

Default CDT remains the best config. multi_kernel=2/3 are worse due to persistent/looped
overhead on this small kernel.

## Remaining Gap (1.17x)

The remaining gap is due to the inherent 2-pass structure:
1. First kernel: row-local reductions (sum over 640 channels per row)
2. Second kernel: channel reductions (sum over 128 batch elements per channel)

The oracle combines both into a single kernel using atomic_add for the channel reductions
(each row atomically accumulates into the channel output). Inductor's generic codegen
cannot merge these two reduction directions into one kernel without atomics.

Closing the remaining gap requires either:
- Atomic accumulator codegen for multi-output reduction scheduling
- Or: recognizing that the row-reduction results are small enough to keep in registers
  while also accumulating the channel reductions (requires multi-dim reduction support)

## Affected Repros

~28 repros have the expand+div+sum pattern (avg_pool2d_backward). This pass benefits all
of them by eliminating the unnecessary spatial expansion.
