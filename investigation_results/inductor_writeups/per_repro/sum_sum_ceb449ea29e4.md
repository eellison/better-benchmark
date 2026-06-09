# sum_sum_ceb449ea29e4

## Classification: SCATTER_ADD_INTO_CAST_VARIANT

## Current Result

- Oracle path: `repros/canonical/sum_sum_ceb449ea29e4/oracle_llama_embedding_scatter_reduce.py`
- Correctness: PASS (compile output closer to eager than oracle)
- Oracle: `206.62 us`
- `torch.compile (with fix)`: `179.26 us`
- Ratio: 0.87x (compile is FASTER than oracle)
- Status: **FIX IMPLEMENTED** - gap fully closed

## Previous Result (before fix)

- Oracle: `208.67 us`
- `torch.compile coordinate_descent_tuning=True`: `459.68 us`
- Ratio: 2.203x

## Root Cause

The existing `scatter_add_into_fusion` pass detects:
```
add(A, index_put(zeros, idx, val, accumulate=True))
```
and rewrites it to `index_put(A, idx, val, accumulate=True)`.

However, this repro has a `convert_element_type` (f32->bf16) between the index_put
and the add:
```
add(A_bf16, convert_bf16(index_put(zeros_f32, [idx], val_f32, accumulate=True)))
```

This prevented the pass from matching, forcing Inductor to:
1. Initialize a full [128256, 2048] f32 zeros buffer (262M elements, ~1GB write)
2. Scatter 2048 rows into it with atomic_add
3. Read the entire 262M element f32 buffer, cast to bf16, add to mm, write output

## Fix

Extended `_find_scatter_add_into_patterns` to also match the convert-wrapped variant.
The rewrite produces:
```
index_put(A_bf16, [idx], convert_bf16(val_f32), accumulate=True)
```

This eliminates:
- The full(0) initialization kernel (262M f32 element write)
- The full read+cast+add kernel (262M elements read + 262M elements write)
- Replaced with: copy mm to output (once) + scatter atomic_add into 2048 target rows

## Kernel Count

| Config | Before | After |
|--------|--------|-------|
| Inductor (before fix) | 4 kernels | - |
| Inductor (with fix) | 3 kernels | copy mm + partial reduction + combo(finalize_sum + scatter) |
| Oracle | 4 kernels | init + row_reduce_scatter + duplicate_fix + finalize |

## Config Exploration Results (with fix)

- multi_kernel=1: ~183 us (ratio 0.87x)
- multi_kernel=2: ~183 us (ratio 0.87x)
- multi_kernel=3: ~183 us (ratio 0.87x)
- coordinate_descent_tuning=True: already enabled in all tests

All configs give the same result - the fix is config-independent.

## Precision Note

The rewrite changes accumulation order for the rare case where multiple source rows
scatter to the same target index. In this repro, only 19 out of 128256 target positions
have 2 contributors (max). The max element-wise difference vs the original is 0.25
(within bf16 precision). The compile output is actually closer to eager than the oracle.

## Status: fix_implemented

## Commit

- Hash: `4427bfd553c` on branch `pr-184905` in `/tmp/pytorch-work`
- File: `torch/_inductor/fx_passes/scatter_reduce_fusion.py`
- Lines: `_find_scatter_add_into_patterns` and `_rewrite_scatter_add_into`

## Speedup

- Before: 459.68 us (2.20x gap)
- After: 179.26 us (0.87x - faster than oracle)
- Improvement: 2.57x speedup
