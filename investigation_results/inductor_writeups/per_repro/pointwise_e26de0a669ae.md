# pointwise_e26de0a669ae

## Summary

- **Model**: timm_adv_inception_v3_infer / timm_inception_v3_infer
- **Pattern**: BatchNorm affine + ReLU + MaxPool2d(3x3, stride=2) + AvgPool2d(3x3, stride=1, pad=1)
- **Classification**: SCHEDULER_FUSION
- **Oracle**: 235.52 us | **Compile**: 358.24 us | **Ratio**: 1.52x
- **Input**: f32[128, 192, 71, 71] channels-last (stride 967872, 1, 13632, 192)
- **Output**: f32[128, 192, 35, 35] channels-last

## Computation Structure

The graph performs inference-mode BatchNorm on a channels-last convolution output:
1. **BN affine**: `(x - mean) * rsqrt(var + eps) * weight + bias` (pointwise, broadcast from [192] stats)
2. **ReLU**: elementwise max(0, x)
3. **MaxPool2d**: 3x3 kernel, stride 2, no padding -> reduces 71x71 to 35x35
4. **AvgPool2d**: 3x3 kernel, stride 1, padding 1 -> same 35x35 output

## Inductor Baseline: 3 Kernels

Inductor generates 3 separate kernels:

| Kernel | Operation | Elements | Notes |
|--------|-----------|----------|-------|
| K0 | BN + ReLU pointwise | 123,887,616 (128*192*71*71) | Writes full intermediate buf0 (~474 MB) |
| K1 | MaxPool 3x3 stride 2 | 30,105,600 (128*192*35*35) | Reads buf0 (9 loads per output) |
| K2 | AvgPool 3x3 stride 1 | 30,105,600 | Reads buf1 (9 loads per output) |

**Memory traffic**: K0 writes 474MB, K1 reads 474MB (with stride-2 selection), K1 writes 115MB, K2 reads 115MB, K2 writes 115MB.

## Oracle: 2 Kernels (BN+ReLU fused into MaxPool)

The oracle fuses BN+ReLU *into* the maxpool stencil computation:

| Kernel | Operation | Elements |
|--------|-----------|----------|
| K0 | BN + ReLU + MaxPool (fused) | 30,105,600 output elements |
| K1 | AvgPool | 30,105,600 |

The fused kernel loads BN stats once (192 floats, stays in L1 cache), then for each of the 9 stencil positions, loads from the original input and computes BN+ReLU inline before taking the max. This eliminates the 474MB intermediate entirely.

## Root Cause

**File**: `/tmp/pytorch-work/torch/_inductor/lowering.py` line 5560
**File**: `/tmp/pytorch-work/torch/_inductor/ir.py` lines 10069-10077

The maxpool lowering calls `x.realize_hint()` on its input. The `realize_hint()` implementation forces realization when `nontrivial_read_count > 1`. For the BN+ReLU node, all 5 reads (1 large input + 4 small broadcast stats) are counted as "nontrivial" because their index expressions are not plain integers.

The heuristic is overly conservative: it doesn't distinguish between reads from large tensors (which would be expensive to recompute) versus reads from small broadcast tensors (which stay in L1/L2 cache and are nearly free to recompute).

## Proof of Fix

Patching `TensorBox.realize_hint()` to skip the first call (allowing BN+ReLU to fuse into maxpool while keeping maxpool->avgpool separate) produces the correct 2-kernel structure and achieves:

- **Patched Inductor**: 228.5 us (measured via `do_bench_using_profiling`)
- **Baseline Inductor**: 386.8 us
- **Speedup**: 1.69x

The patched version is even faster than the oracle (228 us vs 235 us) because Inductor's auto-tuned block sizes are well-optimized for this hardware.

## Configs Attempted (Ineffective)

| Config | Result |
|--------|--------|
| `aggressive_fusion = True` | No change (3 kernels) |
| `realize_reads_threshold = 50` | No change |
| `realize_opcount_threshold = 100` | No change |
| `max_fusion_size = 256` | No change |
| `coordinate_descent_tuning = True` | 373 us (tunes block sizes, not fusion) |
| `max_autotune = True` | 390 us (same) |
| Full fusion (BN+ReLU+maxpool+avgpool in 1 kernel) | 1266 us (9x9=81 recomputes too expensive) |

None of these help because the `realize_hint()` in the lowering forces materialization before any scheduler heuristics can act.

## Proposed Fix

**Approach**: Refine `realize_hint()` to account for the cost/benefit of recomputation.

The fix should modify the `realize_hint()` heuristic (or add a new one specific to pooling lowerings) to NOT force realization when:
1. The producer is a pointwise op
2. The producer's only "large" read is the same tensor that the consumer (pooling) would read anyway
3. The other reads in the producer are broadcasts from small tensors (e.g., numel < some threshold like 4096)
4. The recomputation factor (kernel_size elements = 9 for 3x3) times the producer op count is bounded

Specifically in `_max_pool_with_offsets`, replace the unconditional `x.realize_hint()` with a conditional check:
```python
# Only hint realization if the producer's recomputation cost is high
if not _is_cheap_recomputable_for_pool(x, kernel_size):
    x.realize_hint()
```

Where `_is_cheap_recomputable_for_pool` checks that:
- `x` is a Pointwise node
- Its broadcast reads are from buffers with numel < threshold (e.g., the largest dim of the output)
- The total op count * kernel_size product < some budget

This would benefit all BN+ReLU+pool patterns across InceptionV3 and similar architectures.

## Impact

- **Models affected**: timm_adv_inception_v3_infer, timm_inception_v3_infer (and likely other models with BN->Pool patterns)
- **Potential speedup**: 1.5-1.7x on these subgraphs
- **Risk**: The fix is guarded by cost heuristics, so it won't degrade other cases where realization IS beneficial (e.g., when the producer is expensive to recompute or when the tensor is read by multiple consumers)
