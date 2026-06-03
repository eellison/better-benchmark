# Structured Scatter-Reduce Analysis

## Pattern Description

All four repros implement a common motif from UNet-style networks:

1. **Bilinear upsample backward** (3 repros): `slice -> pad -> bilinear_weight_compute -> 4x index_put(accumulate=True) -> sum_of_4 -> BN_affine_ReLU_mask -> 3x channel_reduction`
2. **Max pool backward** (1 repro): `max_pool_offsets_to_indices -> scatter_add -> mask -> channel_reduction`

The pattern hash differs by spatial resolution (different decoder stages) but all share:
- Scatter operation producing a dense intermediate tensor
- Independent mask applied from a skip connection
- Reduction over ALL scattered dimensions (batch + spatial), keeping only channels

### Repro Details

| Repro | Source Shape | Scatter Output | Channels | Model |
|-------|-------------|---------------|----------|-------|
| sum_sum_sum_f90d684d32cb | [8,128,320,478] | [8,128,160,239] | 128 | pytorch_unet |
| sum_sum_sum_45f02142ecfd | [8,64,640,958] | [8,64,320,479] | 64 | pytorch_unet |
| sum_sum_sum_dadf6aa035dd | [8,256,160,238] | [8,256,80,119] | 256 | pytorch_unet |
| sum_18262b26934c | [32768,3025] scatter_add | [512,64,111,111] | 64 | squeezenet |

## Current Timing from 3-Config Sweep

| Repro | default (us) | combo_persistent (us) | combo_looped (us) | total_bytes | SOL (us) | ratio |
|-------|-------------|----------------------|-------------------|-------------|----------|-------|
| f90d684d32cb | 4435.0 | 3777.7 | 3828.7 | 784,484,956 | 384.7 | 9.8x |
| 45f02142ecfd | 14598.2 | 14496.8 | 14543.9 | 1,570,931,420 | 770.4 | 18.8x |
| dadf6aa035dd | 1534.0 | 1374.3 | 1337.2 | 391,264,028 | 191.9 | 7.0x |
| 18262b26934c | 1413.1 | 1298.5 | 1298.4 | 899,350,788 | 441.1 | 2.9x |

SOL computed at A100 HBM bandwidth (2039 GB/s).

Key observation: combo_persistent/combo_looped provide minimal benefit (0-15%) because the bottleneck is the scatter operation itself, not kernel launch overhead or fusion.

## What the Oracle Does Differently

The existing oracle (`oracle_structured_upsample_reduce.py` for f90d684d32cb) provides:
1. A **torch-direct** implementation that restructures the computation for clarity
2. A **triton-todo** scaffold with detailed design notes for a fused kernel

The oracle's key insight: the computation can be restructured to eliminate the dense scatter intermediate entirely.

## Critical Algebraic Identity

When reducing over ALL dimensions that a scatter writes to:

```
sum(scatter_add(zeros, idx, values), dim=all_scatter_dims)
    == sum(values, dim=all_scatter_dims)
```

The scatter indices become irrelevant because every scattered value ends up in some bin, and we sum all bins. **The scatter is a no-op when followed by a full reduction.**

However, in these repros the mask (from BN-affine + ReLU on the skip connection) breaks this identity because it zeros out some output positions before reduction. This means we cannot simply reduce the source directly.

## Why This Is Hard for Inductor

1. **No scatter-reduce fusion pass exists**: Inductor has `reduced_atomic_contention.py` (partitioned scatter to reduce contention) but nothing that eliminates the scatter when followed by reduction.

2. **The mask breaks simple algebraic optimization**: The `where(relu_gate <= 0, 0, scattered)` prevents the trivial identity from applying. A semantic understanding of the gather/scatter relationship is needed.

3. **Inductor's IR separates scatter from reduction**: The scatter (`index_put` with `accumulate=True`) is lowered as an atomic-write kernel, and the subsequent reduction is a separate kernel. The scheduler has no mechanism to fuse them.

4. **Structured index patterns not exploited**: The bilinear indices follow a regular 2x downsample pattern, but Inductor treats them as arbitrary index tensors. A gather-based formulation requires inverting this mapping.

5. **Atomic scatter is inherently slow**: Each of the 4 `index_put(accumulate=True)` calls performs atomicAdd on a large [N,C,H,W] buffer. For the 45f02142ecfd case, this is 4 * 8*64*320*479 = 312M atomic operations.

## Optimization Strategy: Gather-Based Reduce

Instead of **scatter then mask then reduce**, reformulate as **iterate output positions, gather from source, accumulate**:

```python
# Pseudocode for fused kernel
for each (b, c, h, w) in output_shape:
    # Phase 1: Compute mask from skip connection (independent of scatter)
    mask = compute_bn_relu_gate(skip[b,c,h,w], mean[c], inv_std[c], weight[c], bias[c])
    
    if mask > 0:
        # Phase 2: Gather bilinear contributions from source
        # For 2x bilinear: output[h,w] = sum of 4 weighted source positions
        val = 0
        for (src_r, src_c, w_r, w_c) in bilinear_neighborhood(h, w):
            val += source[b, c, src_r, src_c] * w_r * w_c
        
        # Phase 3: Accumulate into per-channel reduction
        channel_sum[c] += val
        channel_centered_sum[c] += val * (skip[b,c,h,w] - mean[c])
```

### Why This Works

- **No scatter writes at all** (eliminates atomics)
- **No intermediate [N,C,H,W] materialization** (saves 4x output-sized writes + reads)
- **Single pass over output spatial positions** (coalesced reads of skip connection)
- **Structured gather from source** (bilinear pattern is regular, enabling good coalescing)

### I/O Analysis for 45f02142ecfd

| Approach | Traffic | Atomics | Estimated Time |
|----------|---------|---------|---------------|
| Current (scatter-centric) | ~3,427 MB | 312M ops | 14,497 us |
| Optimal (gather-centric) | ~1,570 MB | 0 | 770-2,311 us |

The gather approach reads: source (1,257 MB) + skip (313 MB) = 1,570 MB = total_bytes (SOL-optimal I/O).

Practical coalescing penalty for structured gathers: 1.5-2.5x, giving realistic estimate of 2-3x SOL.

## Possible Implementation Strategies

### Strategy A: FX Pass Pattern Match (Recommended for V1)

Add a new pass in `torch/_inductor/fx_passes/` that recognizes:
```
scatter_add/index_put(accumulate=True) -> [pointwise ops] -> sum(dim=all_scattered_dims)
```
And rewrites to a fused gather-reduce operation.

**Pros**: Clean, modular, pattern-matchable
**Cons**: Needs to handle the mask and intermediate pointwise ops between scatter and reduce

### Strategy B: Custom Triton Template

Register a `TritonTemplate` for the fused scatter-reduce pattern that:
1. Iterates over output positions (or channel tiles)
2. Computes mask inline
3. Gathers from source with structured index arithmetic
4. Performs tree-reduction for channel sums

**Pros**: Maximal performance, full control over tiling
**Cons**: Specific to bilinear/max_pool patterns, needs per-variant templates

### Strategy C: IR-Level Fusion Rule

Teach Inductor's scheduler that when a `ScatterBuffer` is consumed only by `Reduction` nodes (and optional pointwise between), the scatter can be converted to a gather inside the reduction kernel.

**Pros**: Most general, benefits all scatter-then-reduce patterns
**Cons**: Largest implementation scope, needs IR redesign

### Recommended Path

**Phase 1**: FX pass (Strategy A) targeting the specific bilinear upsample backward + BN reduction pattern found in UNet. This covers 3 of 4 repros and the largest gaps.

**Phase 2**: Extend to max_pool backward scatter_add pattern (sum_18262b26934c).

**Phase 3**: Generalize to IR-level rule if more patterns emerge.

## Expected Impact

| Repro | Current Best (us) | Expected After (us) | Savings (us) | Speedup |
|-------|-------------------|--------------------:|-------------:|--------:|
| 45f02142ecfd | 14,497 | 1,500-2,300 | 12,200-13,000 | 6-10x |
| f90d684d32cb | 3,778 | 770-1,150 | 2,600-3,000 | 3.3-4.9x |
| dadf6aa035dd | 1,337 | 380-580 | 760-960 | 2.3-3.5x |
| 18262b26934c | 1,298 | 880-1,100 | 200-420 | 1.2-1.5x |
| **Total** | **20,910** | **3,530-5,130** | **15,780-17,380** | **4.1-5.9x** |

Total absolute savings across all 4 repros: **15,780-17,380 us** per iteration.
This represents the single largest optimization opportunity in the corpus by absolute gap.

## Relation to Existing Inductor Code

- `torch/_inductor/fx_passes/reduced_atomic_contention.py`: Partitions scatter writes to reduce contention. Complementary but does not eliminate the scatter.
- `torch/_inductor/lowering.py:4570` (`index_put`): Current lowering path, uses `index_put_fallback` for accumulate=True cases.
- `torch/_inductor/lowering.py:4861` (`scatter_add`): Lowering for the max_pool case.
- No existing pass eliminates scatter when followed by reduction over scattered dims.
