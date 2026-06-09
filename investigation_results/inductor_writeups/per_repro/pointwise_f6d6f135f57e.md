# pointwise_f6d6f135f57e

## Classification: POINTWISE_CAT_SHARED_COMPUTATION

## Current Result

- Family: `layout_fused_swiglu`
- Oracle path: `repros/canonical/pointwise_f6d6f135f57e/oracle_layout_fused_swiglu.py`
- Correctness: PASS
- Oracle: `26.24 us`
- `torch.compile coordinate_descent_tuning=True`: `25.50 us`
- Ratio: 0.972 (compile is FASTER than oracle)
- Status: `AT_FLOOR` (gap closed)

## Fix Implemented

**Commit**: `d95f59fb1bc` in `/tmp/pytorch-work` (branch `pr-184905`)
**Files**: `torch/_inductor/config.py`, `torch/_inductor/lowering.py`
**Gate**: `config.prefer_concat_kernel_shared_reads_threshold = 6` (default enabled)

The fix improves the `_cat_inputs_share_expensive_reads()` heuristic in the cat
lowering to correctly identify when `pointwise_cat` would duplicate expensive
shared computation across cat branches.

## Root cause

The `pointwise_cat` lowering iterates over the FULL output dimension (16384 x 1536 = 25.2M elements) with conditional branching: for each output element, it checks which cat branch it belongs to and evaluates that branch's computation with masked loads. Both cat branches share expensive intermediate computation (sigmoid/exp), but this computation is independently re-evaluated for each branch, effectively doubling the FLOP count.

The oracle (and ConcatKernel approach) iterates over the INPUT dimension (16384 x 768 = 12.6M elements) and writes TWO outputs per thread. This:
- Halves the iteration count
- Eliminates branch divergence
- Shares intermediate values (sigmoid, exp) naturally

## Before/After

| Metric | Before (pointwise_cat) | After (ConcatKernel) |
|--------|----------------------|---------------------|
| Iteration domain | 25.2M elements | 12.6M elements |
| Stores per thread | 1 (with branching) | 2 (no branching) |
| Shared computation | Duplicated per branch | Computed once |
| Time | ~50 us | ~25.5 us |

## Heuristic Design

The previous heuristic used recursive `get_all_reads()` which found shared graph
inputs even for independent patterns (split->add->cat), causing false positives.

The new heuristic uses `inner_fn_opcount().read_buffers` to check:
1. At least one input has `num_ops >= threshold` (expensive computation)
2. Inputs share >= 2 unique buffer reads (indicating shared computation, not just
   different slices of one input)

Key calibration points:
- Simple `chunk->add->cat`: 3 ops/input, 1 shared buffer -> NOT triggered
- `split->sigmoid/tanh->cat` (independent slices): 6 ops/input, 1 shared buffer -> NOT triggered
- SwiGLU `split->shared_sigmoid->cat`: 10-18 ops/input, 2 shared buffers -> TRIGGERED

## Kernel structure

**Before** (pointwise_cat):
```
triton_poi_fused_...(in_ptr0, in_ptr1, out_ptr0, xnumel=25165824)
  x0 = xindex % 1536
  if x0 < 768:   # branch 1: compute grad
  else:           # branch 2: compute gated
  store(out_ptr0[xindex], result)
```

**After** (ConcatKernel, scheduler-fused):
```
triton_poi_fused_...(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel=12582912)
  # Shared computation: sigmoid, exp
  # Store grad to out_ptr0
  # Store gated to out_ptr1
```

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True, before fix) | 50.78 |
| combo+mk=2 (before fix) | 49.32 |
| Default (cd=True, after fix) | 25.50 |
| Oracle | 26.24 |

## Files modified

- `/tmp/pytorch-work/torch/_inductor/config.py` (line ~1024): New threshold config
- `/tmp/pytorch-work/torch/_inductor/lowering.py` (in `cat()` function): Improved `_cat_inputs_share_expensive_reads()` heuristic
