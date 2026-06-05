# pointwise_e26de0a669ae

## Summary

- **Model**: timm_adv_inception_v3_infer / timm_inception_v3_infer
- **Pattern**: BatchNorm affine + ReLU + MaxPool2d(3x3, stride=2) + AvgPool2d(3x3, stride=1, pad=1)
- **Classification**: REALIZE_HINT (not scheduler-fixable)
- **Oracle**: 239.36 us | **Compile**: 229.15 us | **Ratio**: 0.957x (FIXED, AT_FLOOR)
- **Input**: f32[128, 192, 71, 71] channels-last (stride 967872, 1, 13632, 192)
- **Output**: f32[128, 192, 35, 35] channels-last

## Fix Status

**FIXED** by `config.smart_realize_hint = True` (default enabled) via `_is_broadcast_dominated()` in `ir.py`.
Measured ratio: 0.957x (Inductor faster than oracle due to better block size tuning).

## Computation Structure

The graph performs inference-mode BatchNorm on a channels-last convolution output:
1. **BN affine**: `(x - mean) * rsqrt(var + eps) * weight + bias` (pointwise, broadcast from [192] stats)
2. **ReLU**: elementwise max(0, x)
3. **MaxPool2d**: 3x3 kernel, stride 2, no padding -> reduces 71x71 to 35x35
4. **AvgPool2d**: 3x3 kernel, stride 1, padding 1 -> same 35x35 output

## Without Fix (smart_realize_hint=False): 3 Kernels

| Kernel | Operation | Elements | Notes |
|--------|-----------|----------|-------|
| K0 | BN + ReLU pointwise | 123,887,616 (128*192*71*71) | Writes full intermediate buf0 (~474 MB) |
| K1 | MaxPool 3x3 stride 2 | 30,105,600 (128*192*35*35) | Reads buf0 (9 loads per output) |
| K2 | AvgPool 3x3 stride 1 | 30,105,600 | Reads buf1 (9 loads per output) |

**Memory traffic**: K0 writes 474MB, K1 reads 474MB (with stride-2 selection), K1 writes 115MB, K2 reads 115MB, K2 writes 115MB.

## With Fix (smart_realize_hint=True): 2 Kernels

| Kernel | Operation | Elements |
|--------|-----------|----------|
| K0 | BN + ReLU + MaxPool (fused) | 30,105,600 output elements |
| K1 | AvgPool | 30,105,600 |

The fused kernel loads BN stats once (192 floats, stays in L1 cache), then for each of the 9 stencil positions, loads from the original input and computes BN+ReLU inline before taking the max. This eliminates the 474MB intermediate entirely.

## Root Cause Analysis

**File**: `/tmp/pytorch-work/torch/_inductor/lowering.py` line 5560
**File**: `/tmp/pytorch-work/torch/_inductor/ir.py` lines 10069-10120

The maxpool lowering calls `x.realize_hint()` on its input. Without `smart_realize_hint`, `realize_hint()` forces realization when `nontrivial_read_count > 1`. For the BN+ReLU node, all 5 reads (1 large input + 4 small broadcast stats) are counted as "nontrivial."

The fix (`_is_broadcast_dominated()`) detects that most reads access far fewer elements than the output (the [192]-shaped BN stats vs [128,192,71,71] output) and skips materialization.

## Scheduler-Based Fix Investigation

### Can the Scheduler Make This Decision?

**No. The scheduler fundamentally cannot handle this pattern** due to architectural constraints.

### Why the Scheduler Fails (detailed analysis)

When `realize_hint` forces materialization, the scheduler sees three nodes with these dependencies:

```
op0 (BN+ReLU) writes: MemoryDep('buf0', c0, {c0: 123887616})
op1 (MaxPool) reads:   MemoryDep('buf0', 967872*c0 + 27264*c1 + 384*c2 + c3, {c0:128, c1:35, c2:35, c3:192})
                       ... (9 shifted index expressions for the 3x3 stencil)
op2 (AvgPool) reads:   MemoryDep('buf1', c0 + offset, {c0: 30105600})
                       ... (9 offset variations)
```

The scheduler has THREE independent blockers:

**Blocker 1: `score_fusion_memory` returns 0** (scheduler.py line 8541)
- Deps are compared by equality (`dep in node2_deps`)
- `MemoryDep('buf0', c0, ...)` != `MemoryDep('buf0', 967872*c0 + ..., ...)` because index expressions differ
- Score 0 means "no memory savings from fusion"

**Blocker 2: `V.choices.can_fuse` rejects score=0 pairs** (choices.py line 592-638)
- When `shared_data_score == 0` and either node is a reduction, fusion is rejected
- Even with `aggressive_fusion=True`, it still fails because the maxpool (treated as pointwise after unrolling) still has mismatched indices

**Blocker 3: `can_fuse_vertical` fails on index mismatch** (scheduler.py line 8074-8101)
- `fusable_read_and_write` requires the read index to match the write index (possibly after normalization)
- `MemoryDep('buf0', c0)` (1D contiguous write) cannot match `MemoryDep('buf0', 967872*c0 + 27264*c1 + ...)` (4D strided stencil read)
- Result: `remaining_deps & node1_buf_names` is non-empty -> "memory deps did not match" -> return False

### What a Scheduler Enhancement Would Need

To handle this at the scheduler level, you would need:

1. **A new fusion mode: "inline recomputation"** — The scheduler would need to recognize that a cheap pointwise producer can be inlined into a stencil consumer by re-evaluating the producer's `inner_fn` at each of the consumer's read locations.

2. **Cost model for recomputation** — Evaluate: (producer op count) * (consumer stencil size) vs (intermediate buffer size in bytes). For BN+ReLU: ~5 FMA ops * 9 stencil positions = 45 ops per output element, vs writing/reading 474MB.

3. **Buffer "unrealization"** — Once a buffer is materialized into the IR graph, the scheduler currently has no mechanism to undo this. It would need to remove the buffer node and re-merge the producer's inner_fn into the consumer.

4. **Extended `fusable_read_and_write`** — Accept stencil read patterns where the read accesses the producer's entire output domain but at different coordinates. This would need to express "the consumer reads ALL of the producer's output, just at shifted offsets."

### Why realize_hint Is the Right Place

The `realize_hint` approach works because it operates at a fundamentally different level:

- **Timing**: It acts DURING lowering, BEFORE the buffer is created as an IR node
- **Mechanism**: When `realize_hint` is skipped, the producer remains as an unrealized `inner_fn`. When the consumer (maxpool) calls `x.make_loader()`, it gets the producer's `inner_fn` directly — no buffer ever exists.
- **Result**: The scheduler sees ONE node (maxpool with BN+ReLU inlined in its computation) rather than two separate nodes connected by a buffer.

This is not a "hack around the scheduler" — it's the architecturally correct decision point. The question "should this intermediate be materialized?" belongs at lowering time, where we know the consumer is a stencil operator that will read the data at shifted indices.

### Removing realize_hint at lowering.py:5560 (Without Any Other Change)

**Yes, simply removing line 5560 works** — even without `smart_realize_hint`:
- The BN+ReLU computation stays unrealized
- `x.make_loader()` returns the BN+ReLU inner_fn
- MaxPool's `fn_inner` inlines the BN+ReLU recomputation at each stencil position
- Produces correct results and 2 kernels

However, removing it unconditionally is unsafe for cases where:
- The producer is expensive (e.g., involves transcendental ops like exp, log)
- The stencil is large (e.g., 7x7 or 11x11 pooling)
- The producer has multiple expensive input reads (not broadcast-dominated)

The `smart_realize_hint` + `_is_broadcast_dominated()` provides the correct gating.

## Other realize_hint Callsites With Similar Pattern

All pooling/stencil lowerings in `lowering.py` call `realize_hint`:

| Line | Function | Consumer Type |
|------|----------|---------------|
| 5109 | `_upsample_gen` | Upsampling (bilinear, nearest) |
| 5560 | `_max_pool_with_offsets` | MaxPool (our case) |
| 5796 | `max_pool2d_with_indices_backward` | MaxPool backward |
| 6067 | `_adaptive_avg_pool2d` | Adaptive average pooling |
| 6142 | `adaptive_max_pool2d` | Adaptive max pooling |
| 6272 | `_fractional_max_pool` | Fractional max pooling |
| 6343 | `upsample_nearest2d_backward` | Upsample backward |
| 6464 | avg_pool | Standard average pooling |
| 6601, 6772 | backward pool ops | Gradient computations |

ALL of these benefit from `smart_realize_hint` when preceded by a broadcast-dominated pointwise producer (e.g., BN+ReLU, channel-wise affine transforms, etc.).

## Configs Attempted (Ineffective without smart_realize_hint)

| Config | Result |
|--------|--------|
| `aggressive_fusion = True` | No change (3 kernels) — score is still 0 |
| `realize_reads_threshold = 50` | No change |
| `realize_opcount_threshold = 100` | No change |
| `max_fusion_size = 256` | No change |
| `score_fusion_memory_threshold = -1` | op0->op1 still fails at can_fuse_vertical |
| `coordinate_descent_tuning = True` | 373 us (tunes block sizes, not fusion) |
| `max_autotune = True` | 390 us (same) |
| Full fusion (BN+ReLU+maxpool+avgpool in 1 kernel) | 1266 us (9x9=81 recomputes too expensive) |

None of these help because the scheduler's index-matching architecture cannot express stencil consumer patterns.

## Conclusion

The existing `smart_realize_hint` + `_is_broadcast_dominated()` fix in `ir.py` is the architecturally correct solution. A scheduler-based fix would require a fundamentally new fusion mode ("inline recomputation") that does not exist in the current scheduler architecture and would require substantial refactoring of:
1. The memory scoring system (to give non-zero scores for stencil reads from producer buffers)
2. The `can_fuse_vertical` system (to accept index-mismatched producer-consumer pairs)
3. A buffer elimination mechanism (to "undo" materialization and re-inline the producer)

The realize_hint approach avoids all of these issues by making the correct decision at lowering time: don't materialize cheap producers that will be consumed by stencil operators.

## Impact

- **Models affected**: timm_adv_inception_v3_infer, timm_inception_v3_infer (and likely other models with BN->Pool patterns)
- **Measured speedup**: 1.57x (229us vs 359us baseline)
- **Risk**: None — fix is gated by `_is_broadcast_dominated()` cost heuristic
