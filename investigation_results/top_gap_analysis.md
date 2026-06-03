# Top Gap Analysis: Why combo_kernels+persistent Shows 4-5x Speedup

## Executive Summary

The 4-5x gap between default and combo_persistent in the sweep data stems from **two compounding issues** in the default reduction heuristics:

1. **Missing autotuner configs** (already fixed by pr-184905 stash): The base code only offered R0_BLOCK=1024 for large INNER reductions, while coordinate_descent in the combo config finds R0_BLOCK=32768. The stash adds [2048, 4096, 8192, 16384] to the default config space, reducing this gap from ~4.7x to ~1.3x.

2. **Overly conservative no_split_threshold** (still broken): PR #179729 raised the split reduction threshold from 8K to 524K on Blackwell+ (SM >= 10), which blocks parallelization for batch-norm-backward patterns where xnumel < num_SM. Fixing this gives an additional 1.5x improvement.

## Pattern Characterization

All top-5 repros share the same pattern: **batch-norm backward gradient computation**

| Repro | Model | xnumel (channels) | rnumel (batch*spatial) | Pattern |
|-------|-------|-------------------|----------------------|---------|
| sum_sum_ee85624361a0 | MobileNetV3 | 72 | 401,408 | BN backward (2 sums) |
| sum_2e0e9617102b | ConvNeXtV2 | 80 | 401,408 | add + sum |
| sum_sum_b7f94adef30f | GhostNet | 72 | 401,408 | BN backward (2 sums) |
| sum_6f056653b0d6 | NFNet-F0 | 128 | 294,912 | GELU grad + sum |
| sum_sum_sum_1aa511fcc737 | RepVGG | 96 | 401,408 | 3x BN backward (6 sums) |

Key characteristics:
- Reduction over dims [0,2,3] (batch + spatial), output = [C] channels
- Small xnumel (72-128) relative to GPU SMs (148 on B200)
- Very large rnumel (295K-401K) per output element
- Only 0.5-0.9 waves of thread blocks without splitting

## Root Cause Analysis

### Issue 1: Insufficient Autotuner Config Space (FIXED by stash)

**Location**: `torch/_inductor/runtime/triton_heuristics.py`, `_reduction_configs()` for `ReductionHint.INNER`

**Before stash** (base pytorch): Only `contiguous_config` is tried, which produces R0_BLOCK ~ 1024-2048 for these kernels. With 401K elements reduced per output and only 1024 elements processed per loop iteration, the kernel iterates 392 times through the loop body.

**After stash**: Adds configs with R0_BLOCK in [2048, 4096, 8192, 16384]. The autotuner picks 16384, reducing loop iterations to 25. This alone provides a ~3.2x improvement (237us -> 73us).

**With coordinate descent** (combo config): Finds R0_BLOCK=32768, num_warps=16 for another 1.28x (73us -> 57us).

### Issue 2: no_split_threshold Blocks GPU Utilization (STILL BROKEN)

**Location**: `torch/_inductor/choices.py`, `InductorChoices.reduction_split_factor()`, lines 506-510

```python
no_split_threshold = (
    524288 if props.major is not None and props.major >= 10 else 8192
)
if reduction_numel_hint <= no_split_threshold:
    return 1
```

**Problem**: This threshold (introduced in PR #179729, commit 320c41a1c7c) prevents split reductions for rnumel <= 524K on Blackwell+. For our patterns, rnumel = 295K-401K < 524K, so split is disabled. But with only 72-128 output elements, the GPU has only 72-128 blocks = 0.5-0.9 waves across 148 SMs.

**With split=8**: 640 blocks = 4.3 waves, giving 1.51x additional speedup.

**Why the threshold was raised**: To fix batch=1 entropy over vocab=32K, where split created overhead without benefit (xnumel=1, total work too small).

## Measured Performance (B200, 148 SMs, CUDA Graphs)

### sum_2e0e9617102b (xnumel=80, rnumel=401,408)

| Configuration | Time (us) | vs Default |
|--------------|-----------|-----------|
| Default (no split, R0_BLOCK=16384) | 73.5 | 1.00x |
| Coord descent only (R0_BLOCK=32768) | 57.1 | 1.29x |
| Split=2 (160 blocks, 1.1 waves) | 51.2 | 1.44x |
| Split=4 (320 blocks, 2.2 waves) | 50.2 | 1.46x |
| **Split=8 (640 blocks, 4.3 waves)** | **48.8** | **1.51x** |
| Split=16 (1280 blocks, 8.6 waves) | 48.9 | 1.50x |
| Split=32 (2560 blocks, 17.3 waves) | 55.0 | 1.34x |

### sum_sum_ee85624361a0 (xnumel=72, rnumel=401,408, 2 reductions)

| Configuration | Time (us) | vs Default |
|--------------|-----------|-----------|
| Default (no split) | 255.9 | 1.00x |
| Split=8 | 149.3 | 1.71x |
| Split=8 + coord descent | 148.5 | 1.72x |

### sum_6f056653b0d6 (xnumel=128, rnumel=294,912)

| Configuration | Time (us) | vs Default |
|--------------|-----------|-----------|
| Default (no split) | 137.1 | 1.00x |
| Split=8 | 99.4 | 1.38x |

### Entropy regression guard (xnumel=1, rnumel=32,768)

| Configuration | Time (us) | vs Default |
|--------------|-----------|-----------|
| No split (correct) | 10.3 | 1.00x |
| Split=4 (wrong) | 11.8 | 0.87x (15% regression) |

## Proposed Fix

**File**: `torch/_inductor/choices.py`, method `reduction_split_factor()`

Replace lines 506-510:
```python
no_split_threshold = (
    524288 if props.major is not None and props.major >= 10 else 8192
)
if reduction_numel_hint <= no_split_threshold:
    return 1
```

With:
```python
no_split_threshold = (
    524288 if props.major is not None and props.major >= 10 else 8192
)
if reduction_numel_hint <= no_split_threshold:
    # Allow split bypass when GPU is underutilized AND there is enough total work.
    # This handles BN backward (xnumel=72-128, rnumel=300K-400K) while preserving
    # the no-split behavior for entropy/softmax (xnumel=1, rnumel=32K).
    if reduction_numel_hint * numel_hint < min_elements_per_device:
        return 1  # Not enough total work to benefit from split overhead
    # Fall through to compute split factor (GPU underutilized, enough work)
```

**Guard condition**: `reduction_numel_hint * numel_hint < min_elements_per_device` where `min_elements_per_device = 32 * num_sm * max_threads_per_sm = 9,699,328` on B200.

- Entropy case: 1 * 32768 = 32K < 9.7M -> returns 1 (no split, correct)
- BN backward: 80 * 401408 = 32.1M > 9.7M -> falls through to split (correct)
- Softmax batch=4096: already caught by earlier `numel_hint >= 2*num_sm` check

## Categorization Summary

| Repro | Root Cause | Fix Category |
|-------|-----------|--------------|
| sum_sum_ee85624361a0 | split reduction blocked by no_split_threshold | Threshold/heuristic fix |
| sum_2e0e9617102b | split reduction blocked by no_split_threshold | Threshold/heuristic fix |
| sum_sum_b7f94adef30f | split reduction blocked by no_split_threshold | Threshold/heuristic fix |
| sum_6f056653b0d6 | split reduction blocked by no_split_threshold | Threshold/heuristic fix |
| sum_sum_sum_1aa511fcc737 | split reduction blocked by no_split_threshold | Threshold/heuristic fix |

All 5 repros have the SAME root cause. The fix does NOT require combo_kernels.

## Additional Finding: Autotuner Config Space

The stash already addresses the second factor (insufficient RBLOCK options). Combined:
- Stash RBLOCK fix: recovers 3.2x of the 4.7x gap (pre-stash default -> post-stash default)
- Split reduction fix: recovers additional 1.5x (post-stash default -> split=8)
- Both together: ~4.8x total improvement matches the sweep gap

## What combo_kernels Actually Does Here

For these single-kernel graphs, `combo_kernels=True` has **no effect** (horizontal fusion requires multiple kernels). The `triton.multi_kernel=2` also has **no effect** because the kernel is not classified as persistent_reduction (rnumel > threshold). The entire speedup in the sweep comes from `coordinate_descent_tuning=True` finding larger R0_BLOCK values.
