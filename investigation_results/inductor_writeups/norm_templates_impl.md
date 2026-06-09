# Norm Templates Implementation Investigation (Planck)

## Status: bandwidth_bound_no_improvement

- Queue id: `norm_templates_bn_ln_rms`
- Priority: P1 (rank 5)
- Owner: Planck
- Conclusion: No actionable Inductor optimization found for target repros with current configs.

## Summary

All three target repros were benchmarked under the full set of stash configs
(combo_kernels, multi_kernel, cooperative_reductions, max_autotune_pointwise,
persistent_reductions, nested_reduction, split_reductions) and none show
meaningful improvement over the baseline `coordinate_descent_tuning=True`.

| Repro | Baseline (us) | Best Config (us) | Speedup |
|-------|-------------|-----------------|---------|
| var_mean_765fb8f2c85e (InceptionV3 BN) | 747 | 746 | 1.001x |
| var_mean_598830735cf6 (ResNet152 BN) | 121 | 117 | 1.035x |
| mean_mean_1b98d81214e6 (MT5 RMSNorm) | 27 | 27 | 1.006x |

## Root Cause Analysis: var_mean_765fb8f2c85e

### Kernel Breakdown (profiled)

| Kernel | Time (us) | % Total | Description |
|--------|-----------|---------|-------------|
| triton_red_fused_var_mean_0 | 182 | 24% | var_mean reduction (xnumel=768, rnumel=161312, split=4) |
| triton_per_fused_*_1 | 2 | 0% | Combine 4 splits |
| triton_poi_fused_*_norm_2 | 221 | 30% | Normalization: (x-mean)*rsqrt(var)*w+b+relu |
| triton_poi_fused_*_max_pool_3 | 344 | 46% | _low_memory_max_pool_with_offsets (3x3, stride 2) |
| triton_poi_fused_avg_pool_4 | 128 | 17% | avg_pool2d (3x3, stride 1, pad 1) |

Note: Total (877 us) > measured (747 us) due to L2 cache overlap between kernels.

### Key Dimensions

- Input: [128, 192, 71, 71] f32 = 495.6 MB
- Channels (xnumel outer): 192
- Reduction per channel: 128 * 71 * 71 = 645,248 elements
- Split factor: 4 (xnumel = 768, rnumel = 161,312)
- Normalization output (materialized): [128, 192, 71, 71] with stride padding = 497 MB
- Max pool output: [128, 192, 35, 35] = 120 MB + 30 MB offsets
- Avg pool output: [128, 192, 35, 35] = 120 MB

### Why Combo/Multi Kernel Configs Do Not Help

The combo_kernel and multi_kernel optimizations target **reduction kernels**.
For this repro, the reduction is only 24% of total runtime. The dominant kernels are:

1. **max_pool_with_offsets (46%)**: Reads the 497 MB normalized output through a 3x3
   window and computes both max values and argmax offset indices (i8). The offset
   tracking doubles register pressure and arithmetic vs simple max_pool. This kernel
   is a bandwidth+compute hybrid -- not addressable by reduction-targeting configs.

2. **Normalization pointwise (30%)**: Reads 495 MB input + small buffers, writes 497 MB.
   Achieves 4485 GB/s effective bandwidth (via L2 cache reuse from the preceding
   reduction which just read the same input). Already near-optimal.

### Why Fusion Opportunities Are Blocked

- **Reduction + Normalization**: Requires 2-pass (must finish reduction before
  normalization starts). Could save kernel launch overhead but not memory traffic
  since the 495 MB input doesn't fit in L2 (50 MB on H100).
  
- **Normalization + MaxPool**: Blocked because buf6 (normalized output) has 2 consumers
  in the IR (max pool values + max pool offsets). Inductor correctly materializes it
  to avoid 9x recomputation (each max pool output reads a 3x3 window = 9 elements).
  
- **MaxPool + AvgPool**: Different output dimensions prevent direct fusion.

### Memory Bandwidth Analysis

Total kernel I/O (sum of all reads + writes): ~2373 MB
Measured time: 747 us
Effective throughput: 2373 MB / 747 us = 3.17 TB/s
Device DRAM BW: ~2.0-2.6 TB/s

The >100% DRAM utilization confirms significant L2 cache reuse between sequential
kernels (normalization output stays partially in L2 for max pool reads).

## Configs Tested

| Config | Result |
|--------|--------|
| coordinate_descent_tuning=True (baseline) | 747 us |
| + combo_kernels + per_subkernel_blocks + multi_kernel=3 | 746 us (1.001x) |
| + max_autotune + max_autotune_pointwise | 747 us (1.001x) |
| + triton.multi_kernel=3 (alone) | 746 us (1.001x) |
| + triton.nested_reduction=True | 747 us (no change) |
| + triton.force_cooperative_reductions=True | FAIL (too many blocks) |
| + triton.persistent_reductions=False | 747 us (no change) |
| split_reductions=False | 913 us (1.22x SLOWER) |

## Oracle Comparison

The existing oracle scaffold (oracle_bn_training_forward.py) only covers the
BN+ReLU portion (excluding pooling). Its timing:

- Oracle Triton (BN+ReLU only): 522 us
- Inductor (BN+ReLU only, from profiling): 405 us

The Inductor-generated code is FASTER than the oracle for the BN portion because
the oracle uses a naive single-block-per-channel approach with elems_per_channel=645K
(requires huge register tiles), while Inductor's 4-way split reduction achieves
better SM occupancy and memory access coalescing.

## What Would Actually Help (Future Work)

1. **Fused normalization + max_pool kernel** (estimated 1.3-1.5x): A custom Triton
   template that computes normalization inline during the max pool computation,
   eliminating the 497 MB intermediate. This requires:
   - Tiled computation where a strip of normalized rows fits in shared memory
   - The max pool then operates on the strip without going through DRAM
   - Implementation complexity: HIGH (custom template with shared memory staging)

2. **Two-pass fused kernel** (estimated 1.05-1.1x): A single kernel launch that
   does pass 1 (reduction) then pass 2 (normalization) with improved L2 locality
   between passes. Current Inductor can't express this.

3. **Better max_pool_with_offsets codegen** (estimated 1.1x): The offset tracking
   logic generates ~130 lines of comparisons. A more compact argmax implementation
   using tl.where chains or custom reduction primitives could reduce register pressure.

## Recommendation

Mark as **bandwidth_bound** -- the available Inductor config knobs (designed for
reduction optimization) do not address the actual bottleneck (large pointwise +
pooling materializations). The 1.035x improvement seen on var_mean_598830735cf6
is within noise.

Meaningful improvement requires structural changes:
- A BN+pool fused template (P2 complexity)
- Or splitting the pattern into separate BN and pool optimizations

The existing oracle scaffold already demonstrates that the BN portion alone is
well-optimized by Inductor (even faster than the hand-written oracle).
