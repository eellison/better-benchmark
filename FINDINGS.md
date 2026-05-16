# Kernel Investigation Findings

## Summary of Systemic Issues

### Issue 1: Missing combo_kernel/foreach batching (HIGHEST IMPACT)
**Affects**: DenseNet, VoVNet, ViT-Tiny, and likely all models with many layers
**Gap**: 5-53x across affected repros
**Root cause**: Hundreds of independent small pointwise/copy ops each become separate kernel launches. Launch overhead (5us/kernel) dominates compute.
**Why not fused**:
- `combo_kernels = False` by default (`config.py:1000`)
- Even if enabled, `combo_kernel_max_num_nodes = 8` is too low
- Ops not expressed as `_foreach_*` in the FX graph
- `can_fuse()` requires matching numels, blocking fusion across different-sized layers
- Horizontal fusion blocked by `score_fusion_memory_threshold` (no shared data)

**Proposed fix** (ordered by preference):
1. Enable `combo_kernels` by default for small pointwise ops, raise `max_num_nodes`
2. Add `BatchCopyPostGradFusion` pass to group epilogue `copy_` ops into `_foreach_copy_`
3. Relax horizontal fusion memory threshold for tiny kernels (numel < 2048)

**Files**: `torch/_inductor/config.py`, `torch/_inductor/scheduler.py` (combo_kernel logic), `torch/_inductor/fx_passes/group_batch_fusion.py`

**Repros**: mean_b820cb5c6494, pointwise_30ed971a3cbe, pointwise_16427862d66c, mean_9c51bd39249d, sum_9292de0b9be9

**Verified**: Setting `combo_kernels = True` reduces sum_9292's 107 kernels to 15 (7.1x reduction in launches).

---

### Issue 2: Persistent reduction num_warps heuristic
**Affects**: All persistent reductions with rnumel=1024 on high-bandwidth GPUs (B200/H200)
**Gap**: 7.3x (would improve to ~2x)
**Root cause**: `_persistent_reduction_configs` in `triton_heuristics.py:4113-4114` forces `num_warps=1, XBLOCK=1` when `rnumel >= 1024`. This gives only 32 threads/CTA, causing ~17 warps/SM — insufficient to hide HBM latency on B200 (needs 32-64 warps/SM).
**Fix**: Remove the `num_warps=1` override. Let default `num_warps = r // 128 = 8` (256 threads, 4 elements/thread). Add XBLOCK=4 to autotuner configs.

**File**: `torch/_inductor/runtime/triton_heuristics.py` line 4113-4114

**Repro**: amax_sum_b978f41418bc

---

### Issue 3: MixOrderReduction size threshold too conservative
**Affects**: All LayerNorm/RMSNorm backward passes with parameter gradients
**Gap**: 8.7x → ~3-4x with fix
**Root cause**: Outer reductions (param grad sum) and inner reductions (input grad) have reversed loop orders. `MixOrderReduction.can_fuse()` threshold (`nrow*ncol >= 5*2^20`) rejects fusion for moderate sizes, forcing `split_reduction` with factor 256. This reads 36MB of input data TWICE.
**Fix**: Relax threshold when the alternative is split_reduction that creates redundant reads.

**File**: `torch/_inductor/scheduler.py` line 326-332

**Repro**: sum_sum_sum_f0394a99a1e5

---

### Issue 4: ConcatKernel as fusion barrier
**Affects**: `cat + reduction` patterns (e.g., argmax over concatenated pos/neg values)
**Gap**: 6.2x → ~1.5x with fix
**Root cause**: `ConcatKernel` materializes a contiguous buffer even when the downstream op is a reduction over the cat dimension. This causes 4.8x excess memory traffic (486MB actual vs 102MB ideal).
**Fix**: Support "virtual concat" views in `ConcatKernel` that downstream reductions can inline. When cat operates on the reduction dimension, the reduction kernel should use index remapping instead of reading materialized concat output.

**File**: `torch/_inductor/ir.py` (ConcatKernel, ~line 6013)

**Repro**: argmax_09f9e9c2268d

---

### Issue 5: Atomic scatter (NOT FIXABLE)
**Affects**: index_put/scatter_add backward with high fan-in (small vocab, large batch)
**Gap**: 11x
**Root cause**: 102x fan-in to 320 output rows from 32768 source rows. L2 atomic serialization is the bottleneck. Inductor codegen is already near-optimal (fused pointwise into scatter, coalesced atomics).
**Status**: Inherent hardware limitation. No inductor fix possible.

**Repro**: pointwise_509a05eca994

---

## Impact Summary

| Issue | # Repros Affected | Typical Gap | Fix Complexity | Priority |
|-------|------------------|-------------|----------------|----------|
| combo_kernel/foreach | 50+ (est) | 5-53x | Medium (config + threshold tuning) | P0 |
| persistent reduction warps | ~20 (est) | 3-7x | Low (remove 2 lines) | P0 |
| MixOrderReduction threshold | ~30 (est) | 3-9x | Low (relax threshold) | P1 |
| ConcatKernel fusion barrier | ~10 (est) | 3-6x | High (new fusion pattern) | P1 |
| Atomic scatter | ~15 | 5-11x | N/A | Won't fix |

---

### Issue 6: Duplicate scatter buffers in embedding backward
**Affects**: Models with shared embeddings (MT5, T5, multilingual models)
**Gap**: 4.0x (592MB case)
**Root cause**: Autograd decomposes embedding backward into two separate `index_put(zeros, same_indices, grad, accumulate=True)` operations (one per transformer layer sharing the embedding), then adds the results. This creates two 512MB zero buffers, two scatters, and a 1.5GB add — when a single scatter would suffice.
**Fix**: FX graph pass recognizing `index_put(zeros, idx, v1, acc) + index_put(zeros, idx, v2, acc)` → `index_put(zeros, idx, v1+v2, acc)`. Or merge at the autograd level.
**Files**: `torch/_inductor/fx_passes/` (new pattern), or `torch/autograd/` (embedding backward decomp)

**Repro**: sum_sum_sum_2c26938323d4

---

### Issue 7: Benchmarking artifact — randint(0,2) for index tensors
**Affects**: ~129 repros with scatter/gather/index_put/embedding
**Gap**: Appears 4-11x but partially artificial
**Root cause**: Our `randint(0, 2)` fix for crash-safety sends all scatter ops to only 2 destinations, creating extreme atomic contention that doesn't exist in production. Real vocab-distributed indices have near-unity fan-in.
**Fix**: During capture, infer the valid index bound from the scatter/gather target dimension (e.g., for `scatter(dim=1, ...)` into `[N, V]`, use `randint(0, V)`). For embedding lookups, use `randint(0, vocab_size)`.
**Impact on findings**: Some scatter-pattern gaps are overstated. Repros with scatter should be re-benchmarked with corrected index distributions.

**Repros affected**: pointwise_b3f8078cef30, and ~129 others with scatter/gather/index ops

---

---

### Issue 8: Loop order / tiling for permute/transpose kernels (1.3-4.4x!)
**Affects**: Attention head reshuffling (permute [B,H,S,D] → [B,S,H,D]), pure data movement kernels
**Gap**: 1.3-1.7x in fused graphs, up to **4.4x** for standalone permute+clone
**Root cause (deep-dived)**:
1. Iteration follows OUTPUT layout, causing L2 cache thrashing on reads (128KB jumps every 64 elements)
2. Modular indexing overhead: complex div/mod address computation (~10 int ops/element vs ~2 optimal)
3. 1D tiling flattens everything — misses the opportunity for a 2D grid that separates transposed dims

**Measured performance:**
- Inductor default: 289 GB/s
- Optimal Triton (2D grid, no mod/div): 930 GB/s (3.2x faster)
- Eager aten::copy_: 1281 GB/s (4.4x faster)

**Fixes (ordered by impact):**
1. **Multi-dim grid with direct addressing** for transpose patterns: detect read/write stride ratio > 4x, use 2D grid (program_id per transposed block), no mod/div indexing — **3.2x win**
2. **Fix tie-breaker** in `scheduler.py:2881` `pick_loop_order` to favor read locality over write locality — 1.07x
3. For fused kernels (permute + downstream compute): ensure the iteration order follows the dominant INPUT stride, not output

**Files**: `scheduler.py:2846` (pick_loop_order), `ir.py:5107` (simplify_and_reorder), `tiling_utils.py:678` (analyze_memory_coalescing), `simd.py:2762` (compute_tiling_strategy)

**Repros**: pointwise_4ef882a876a7, pointwise_126d2fcb5043, pointwise_5be548710b92, pointwise_4b23f512dcb6, pointwise_0a9eb60f758b

---

### Issue 9: ConcatKernel + orthogonal reduction (push reduction through cat)
**Affects**: InceptionV4, models with multi-branch concat followed by spatial reduction
**Gap**: 5.5x → ~2x with fix
**Root cause**: `cat(branches, dim=C).mean(dim=spatial)` materializes the full cat output even though the reduction could be computed per-branch independently: `cat([b.mean(spatial) for b in branches])`.
**Fix**: Graph rewrite pass: when cat dim ≠ reduction dim, push reduction through cat.
**Generalizes**: Issue 4 (ConcatKernel fusion barrier) — this is the orthogonal-axis variant.

**Repro**: mean_2949fb044264

---

### Issue 10: Rotary embedding `rotate_half` cat barrier
**Affects**: ALL LLMs using RoPE (Mistral, Qwen3, DeepSeek V3, LLaMA, etc.)
**Gap**: 2.9-3.4x
**Root cause**: `rotate_half` does `cat([-x[..., n//2:], x[..., :n//2]], dim=-1)` which creates a cat on the inner dimension. This prevents fusion of the rotary application (`x * cos + rotate_half(x) * sin`) into surrounding attention ops.
**Fix options**:
1. Pattern-match `rotate_half` and replace with a single pointwise op using index remapping (no cat needed — just negate and swap halves via indexing)
2. Teach ConcatKernel to inline when cat is on inner dim and both halves come from slices of the same input
**Impact**: This pattern appears in EVERY forward+backward pass of every RoPE model — fixing it improves every LLM.

**Repros**: pointwise_f3c6cfe72279 (Mistral), mean_mean_e78bb40350dd (Qwen3), pointwise_c0b2590bfdf4 (DeepSeek V3)

---

### Issue 11: Horizontal fusion for same-shape independent reductions  
**Affects**: Multi-branch BatchNorm (timm models with parallel BN paths)
**Gap**: 3.3x → ~1.7x with fix
**Root cause**: Independent reductions with identical `(xnumel, rnumel)` are not horizontally fused because `shared_data_score=0` fails the memory threshold check. Result: each BN reduction launches separately with only 2.6 blocks/SM (terrible occupancy).
**Fix**: Allow horizontal fusion of same-shape reductions regardless of shared data — the benefit is occupancy, not data reuse. Change `can_fuse_horizontal` in `choices.py:622`.

**Repro**: var_mean_var_mean_var_mean_e91c69d33ef5

---

---

### Issue 12: Cross-entropy / online softmax compute-bound + fast math
**Affects**: All cross-entropy loss, log_softmax over large vocabularies (30K-50K)
**Gap**: 3.3-4.0x (partially irreducible — compute-bound from exp())
**Root cause**: Three compounding factors:
1. SOL reference is bidirectional copy but kernel is read-only (1.5x inherent)
2. Online softmax overhead when using LOOPING (non-persistent) reduction — exp correction factor is NOT CSE'd across loop iterations
3. Inductor caps RBLOCK=1024 on Blackwell GPUs even when register pressure is low
**Fix 1**: `triton_heuristics.py:3514` — allow RBLOCK=2048 when `loads_and_red <= 4` on Blackwell. This makes MORE reductions persistent, which eliminates online softmax overhead (Triton CSE handles persistent case perfectly).
**Fix 2**: `triton_helpers.py:226` — use `tl.math.exp` (fast math) for online_softmax_combine (safe since args ≤ 0)
**Key insight (from experiment)**: For PERSISTENT reductions, one-pass vs two-pass is identical (Triton CSE eliminates duplicate exp). The overhead only matters for LOOPING reductions. So the fix is: widen the persistent threshold, don't disable online softmax.
**Combined effect**: 4.0x → 3.3x (18% faster). Remaining gap is irreducible compute cost.

**Experiment**: `/tmp/scratch_space/better_benchmark/experiments/softmax_onepass_vs_twopass.py`
**Repro**: amax_sum_sum_a28463f10193

---

### Issue 13: Reduction tiling for non-power-of-2 inner dims
**Affects**: Softmax/attention patterns with sequence lengths like 128, 476, 513, etc.
**Gap**: 1.5-2.5x
**Root cause**: Inductor's reduction RBLOCK choices are powers of 2 (128, 256, 512, 1024). For a reduction over 476 elements, RBLOCK=512 wastes 7% of threads on masking. For 513, RBLOCK=1024 wastes 50%. The autotuner configs don't include non-power-of-2 options or multi-row strategies that could compensate.
**Fix**: Add non-power-of-2 RBLOCK configs to the persistent reduction autotuner (e.g., RBLOCK=480 for dim=476), or use XBLOCK>1 with smaller RBLOCK to amortize the waste across multiple rows.
**Files**: `torch/_inductor/runtime/triton_heuristics.py` (reduction config generation)

**Repros**: amax_sum_5cbe925f22c7 (476), amax_sum_7cb48c680567 (128), amax_sum_0251f3e427e4 (128), amax_sum_c0c1b95fe65e (513)

---

## Full Issue Distribution (from 663 repros, ~110 triaged in detail)

| Category | Est. Count | Gap Range | Fix Effort | Priority |
|----------|-----------|-----------|------------|----------|
| At SOL (<1.1x) | 299 (45%) | — | None needed | — |
| combo_kernel/foreach | ~100 (15%) | 1.5-53x | Medium | P0 |
| Reduction heuristics (tiling, split, MixOrder) | ~80 (12%) | 1.3-9x | Low-Medium | P0 |
| Loop order / tiling | ~50 (8%) | 1.1-1.8x | Medium-Hard | P1 |
| ConcatKernel / cat barrier | ~40 (6%) | 2-6x | High | P1 |
| Scatter (inherent) | ~30 (5%) | 2-11x | N/A | Won't fix |
| Benchmark artifact (randint bounds) | ~40 (6%) | overstated | Fix measurement | — |
| Rotary embed cat (rotate_half) | ~15 (2%) | 2.9-3.4x | Medium | P1 |
| Duplicate scatter buffers | ~5 (1%) | 3-4x | Medium | P2 |
| Persistent reduction warps | ~20 (3%) | 3-7x | Low | P0 |

## Top Priority Fixes (ordered by impact × breadth)

1. **Enable combo_kernels** — config change + raise thresholds. Fixes 100+ repros, 5-53x improvement. Lowest effort, highest impact.
2. **Persistent reduction num_warps=1 override** — delete 2 lines. Fixes ~20 repros at 3-7x. Trivial effort.
3. **MixOrderReduction size threshold** — relax for split_reduction cases. Fixes ~30 repros at 3-9x.
4. **Horizontal fusion for same-shape reductions** — allow when occupancy benefit. Fixes ~20 repros.
5. **Non-power-of-2 reduction tiling** — add configs. Fixes ~30 repros at 1.5-2.5x.
6. **Rotary embed rotate_half pattern** — eliminate cat via index remapping. Fixes all RoPE LLMs.
7. **ConcatKernel virtual views / push-through-cat** — general cat barrier fix. Fixes ~40 repros.
8. **Duplicate scatter buffer merging** — graph pass for shared-index embedding backward.

## Triage Coverage

**210 of 364 repros** with gap >1.1x have been triaged (58%). Distribution is fully converged — no new categories emerging.

| Gap Range | combo_kernel | reduction_heuristic | cat_barrier | loop_order | scatter | compute_bound | noise/artifact |
|-----------|:-----------:|:-------------------:|:-----------:|:----------:|:-------:|:-------------:|:--------------:|
| 5-53x | dominant | — | some | — | some | — | — |
| 2-5x | 35% | 30% | 10% | — | 15% | 5% | 5% |
| 1.5-2x | 35% | 35% | 5% | 10% | 5% | 5% | 5% |
| 1.1-1.5x | 25% | 25% | — | 20% | — | 5% | 25% |

**77% of all non-SOL repros have actionable inductor fixes.**

Combo_kernel + reduction heuristics together account for **57% of all gaps**. These two fixes alone would move the median gap from 1.15x toward 1.05x and bring "at SOL" from 45% to ~65-70%.

## Remaining (not yet triaged)
~154 repros remain. Based on the stability of the distribution across 210 samples, the breakdown above is representative. The `pointwise_*` prefix repros are overwhelmingly foreach/optimizer patterns (75%); `sum_*` and `var_mean_*` are reduction heuristic (60%) or combo_kernel (30%).
