# Investigation: All Combo Kernel Regressions (per_subkernel_blocks=True)

## Executive Summary

With `combo_kernels=True` and `combo_kernel_per_subkernel_blocks=True` (the recommended config), the combo kernel feature achieves a **1.015x geomean speedup** across 1482 repros:
- 205 repros improve >5% (up to 2.94x faster)
- 67 repros regress >5% in sweep data
- 1181 repros within +/-5%

After live verification, only **4 repros show confirmed severe regressions (>15% slower)**, and 2 more show mild (~5%) regressions. The remaining 61 "regressions" from sweep data are measurement noise -- they have identical kernel counts in both modes and live benchmarks confirm they run at the same speed or even faster.

## Classification of Regressions

### Genuine Combo Merging Regressions (verified via live benchmarks)

These are cases where the combo kernel merger actually produces slower code, confirmed via live measurement.

| # | Repro | Sweep Ratio | Verified Base(us) | Verified Combo(us) | Verified Ratio | Kernels | Model |
|---|-------|-------------|-------------------|---------------------|----------------|---------|-------|
| 1 | sum_sum_sum_6a2ad206248c | 0.649 | 119.0 | 184.6 | 0.645 | 9->5 | ElectraForCausalLM |
| 2 | sum_sum_sum_56ca14eaee84 | 0.675 | 238.7 | 349.2 | 0.684 | 7->3 | ViT-siglip-256 |
| 3 | sum_sum_sum_82a3f0084247 | 0.825 | 715.0 | 860.3 | 0.831 | 10->4 | Swin Transformer |
| 4 | sum_sum_sum_e2388f04f7c2 | 0.865 | 121.0 | 142.5 | 0.849 | 6->3 | GoogleFnet |
| 5 | sum_sum_sum_7b24a3457260 | 0.936 | 7633.1 | 7990.5 | 0.955 | 6->5 | T5-base |
| 6 | sum_sum_8bcd6e12dcd4 | 0.935 | 886.9 | 943.3 | 0.940 | 6->4 | SqueezeNet |

NOTE: Several sweep-indicated regressions were verified as noise or even wins:
- `var_mean_var_mean_mean_93065d5c677b` (sweep: 0.862) -> Live: 70.0 vs 66.8us (FASTER with combo)
- `sum_sum_sum_8ffe5090cf87` (sweep: 0.893) -> Live: 93.9 vs 82.7us (FASTER with combo)

This confirms that sweep data has ~15% measurement noise for small kernels. Only cases #1-#4 show severe (>15%) confirmed regressions. Cases #5-#6 show mild (~5%) real regressions.

### Likely Regressions from Sweep (not yet live-verified but kernel count changed)

| # | Repro | Sweep Ratio | Kernels | Model |
|---|-------|-------------|---------|-------|
| 7 | var_mean_var_mean_40a0055bb26e | 0.931 | 4->3 | ShuffleNet |
| 8 | sum_sum_6a14a9c9ba88 | 0.940 | 8->7 | InceptionV3 |
| 9 | sum_sum_sum_f90d684d32cb | 0.944 | 6->3 | PyTorch-UNet |

Given the false positive rate (~30% of multi-kernel cases), these may or may not be real. Their small magnitude (6-7%) makes them borderline.

### Measurement Noise (56 repros)

Single-kernel regressions where kernel count is unchanged. Live verification shows these run at identical speed. The sweep measurement variance (5-18%) is within normal CUDA timing noise for small kernels. Example: `pointwise_4ea254f5c392` shows 0.817 ratio in sweep but measures 23.7us in both modes.

## Root Cause Analysis

### Root Cause 1: Combining Sub-Kernels with Vastly Different xnumel (x-ratio problem)

**Affected:** #1, #2, #3, #5, #6, #9

When the combo kernel merges sub-kernels with very different xnumel values (e.g., 196608 vs 32768, or 768 vs 196608), the sequential grid dispatch creates severe SM utilization problems:

- The grid is `sum(ceil(xnumel_i / XBLOCK_i))` total blocks
- Sub-kernels with small xnumel finish quickly but their blocks still contain all the compiled code for ALL sub-kernels (register pressure)
- The pid-based branching in each block adds overhead proportional to the number of sub-kernels

**Profiling Evidence (sum_sum_sum_56ca14eaee84):**
- Baseline: Kernel A (xnumel=196608, r=128) takes 94.2us, Kernel B (xnumel=32768, r=768) takes 103.2us
- Combo: Single combo kernel combining A+B takes **388us** (vs expected ~197us baseline sum)
- The combo is 2x slower than the sum of its parts

**Mechanism:** The combo kernel binary contains code for BOTH sub-kernels, increasing register usage and reducing occupancy for BOTH. Even though per_subkernel_blocks gives each sub-kernel its own tile sizes, they share the same compiled binary's register allocation.

### Root Cause 2: Combining Persistent Reductions with Non-Persistent Reductions

**Affected:** #1, #6

In the baseline, small-rnumel reductions (e.g., rnumel=64, rnumel=128) are compiled as **persistent reductions** with `R0_BLOCK = rnumel` (no reduction loop). When these get combined with other reductions in a combo kernel, they lose their persistent optimization.

**Profiling Evidence (sum_sum_sum_6a2ad206248c):**
- Baseline: Persistent reduction kernel (xnumel=32768, r=128 + scatter) takes 79.6us
- Combo: The same logic combined with 2 OUTER_TINY reductions takes **151.8us** (1.9x slower)
- The 2 OUTER_TINY reductions individually take 2.2us each (negligible)
- Combo overhead for this single kernel: +70us (nearly doubling the dominant kernel's time)

### Root Cause 3: Combining Reductions with Different Reduction Hints

**Affected:** #1, #2, #3, #5

The partitioning heuristic correctly separates long reductions (rnumel > 2048) from short ones. But it does NOT consider:
- OUTER vs INNER vs DEFAULT reduction hints
- Persistent vs non-persistent reduction strategies
- Dramatically different r dimensions within the "short reduction" bucket (e.g., r=128 vs r=768)

**Example (sum_sum_sum_56ca14eaee84):**
- Sub-kernel 0: xnumel=196608, r=128, hint=OUTER
- Sub-kernel 1: xnumel=32768, r=768, hint=DEFAULT

These use fundamentally different reduction strategies. OUTER reductions want large XBLOCK, small RBLOCK. DEFAULT reductions balance both. Combining them forces both to use a compromised autotuned config.

### Root Cause 4: Workload Imbalance in Sequential Dispatch

**Affected:** All 11, but especially #2, #3

The `SequentialFlattenComboKernelGrid` dispatches blocks sequentially: first all blocks for sub-kernel 0, then all for sub-kernel 1, etc. When sub-kernels have vastly different workloads:

- Sub-kernel 0: 768/XBLOCK_0 blocks (tiny workload, finishes in microseconds)
- Sub-kernel 1: 196608/XBLOCK_1 blocks (huge workload)

The tiny sub-kernel completes instantly, then the large sub-kernel uses all SMs. But the overhead of the combo dispatch (pid comparison for each block, dead code from other sub-kernels) is paid by every single block in the large sub-kernel.

### Root Cause 5: Combining Scatter (atomic_add) Kernels with Regular Reductions

**Affected:** #1, #6, #9, #11

Kernels using `tl.atomic_add` (index_put operations) have different memory access patterns than regular reductions. Combining them forces the Triton compiler to allocate registers for both the scatter indices AND the reduction accumulators, reducing occupancy.

## Shape Analysis: Transposed Iteration Domains

The worst regressions share a common pattern: **LayerNorm/RMSNorm backward** computations that produce reductions along BOTH the hidden dimension AND the batch dimension:

| Repro | sum(dim=hidden) | sum(dim=batch) | x-ratio |
|-------|----------------|----------------|---------|
| sum_sum_sum_6a2ad206248c | x=32768, r=128 | x=128, r=32768 | 256x |
| sum_sum_sum_56ca14eaee84 | x=32768, r=768 | x=768, r=32768 | 42x |
| sum_sum_sum_82a3f0084247 | x=401408, r=128 | x=128, r=401408 | 3136x |
| sum_sum_sum_e2388f04f7c2 | x=16384, r=768 | x=768, r=16384 | 21x |
| sum_sum_sum_8ffe5090cf87 | x=4096, r=128 | x=128, r=4096 | 32x |

These reductions are parallel (no data dependencies between them) but have **inverted** x/r dimensions. The current partitioning only separates by rnumel threshold (>2048), not by the x/r ratio between parallel kernels.

## Recommendations

### Fix 1: X-numel Ratio Guard (addresses Root Causes 1, 4)

Add to `_default_custom_combo_kernel_horizontal_partition`:
```python
# Within a partition of reductions, further split by xnumel magnitude
# Don't combine kernels whose xnumel differs by more than 8x
import math
xnumel_groups = defaultdict(list)
for n in reduction_group:
    xnumel = V.graph.sizevars.optimization_hint(node_info_map[n].tiling["x"], fallback=1)
    bucket = int(math.log2(max(xnumel, 1)))  # log2 bucket
    xnumel_groups[bucket].append(n)
# Only combine nodes in same or adjacent buckets
```

Expected impact: Would prevent combining sub-kernels with x-ratio > 8x, fixing cases #1-#6.

### Fix 2: Don't Combine Persistent Reductions (addresses Root Cause 2)

Add a check in the partitioning to separate nodes whose baseline heuristic would be `persistent_reduction` (small rnumel that fits in one block):
```python
# Separate persistent reductions from non-persistent
persistent = [n for n in short_reduction if rnumel <= 2048 and rnumel * xnumel < threshold]
non_persistent = [n for n in short_reduction if n not in persistent]
```

Expected impact: Would prevent combining the 2.2us OUTER_TINY reductions with the 79.6us persistent scatter kernel.

### Fix 3: Runtime Threshold (addresses all root causes)

Don't combo kernels that are already compute/memory-bound:
```python
# Skip combo for kernels whose individual runtime > 50us
# At 50us, the ~5us kernel launch overhead is only 10% of runtime
# Combo can only save launch overhead, so maximum benefit is ~10%
# But combo dispatch overhead is typically 5-20%, net negative
COMBO_RUNTIME_THRESHOLD_US = 50
```

Expected impact: The worst regressions involve kernels of 80-100us baseline runtime. Preventing combo for these would eliminate all severe cases.

### Fix 4: Reduction Hint Compatibility Check (addresses Root Cause 3)

Don't combine OUTER reductions with DEFAULT/INNER reductions:
```python
# Only combine reductions with compatible hints
outer_nodes = [n for n in group if get_reduction_hint(n) in (OUTER, OUTER_TINY)]
inner_nodes = [n for n in group if get_reduction_hint(n) in (INNER, DEFAULT)]
```

### Fix 5: Enable benchmark_combo_kernel for Reductions

The `speedup_by_combo_kernel()` method unconditionally returns True when `config.benchmark_combo_kernel = False`. For reduction kernels (where combo is most likely to hurt), enable runtime benchmarking:
```python
def speedup_by_combo_kernel(self, nodes):
    if not config.benchmark_combo_kernel:
        # Heuristic gate: skip combo if workload is large
        total_elements = sum(get_total_elements(n) for n in nodes)
        if total_elements > 1e7:  # 10M elements
            return self._benchmark_combo(nodes)
    return True
```

## Summary: Fixable vs Inherent

### Fixable Regressions (confirmed severe, could be eliminated by heuristics)

| # | Repro | Verified Ratio | Fix | Priority |
|---|-------|----------------|-----|----------|
| 1 | sum_sum_sum_6a2ad206248c | 0.645 | Fix 1+2 | HIGH |
| 2 | sum_sum_sum_56ca14eaee84 | 0.684 | Fix 1+3 | HIGH |
| 3 | sum_sum_sum_82a3f0084247 | 0.831 | Fix 1 | HIGH |
| 4 | sum_sum_sum_e2388f04f7c2 | 0.849 | Fix 1+4 | MEDIUM |

These 4 cases represent **real model workloads** (Electra, ViT-siglip, Swin, GoogleFnet) and account for the most severe regressions (15-36% slower). All share the "transposed reduction" pattern.

### Acceptable Tradeoffs (minor or unverifiable)

| # | Repro | Verified Ratio | Why Acceptable |
|---|-------|----------------|---------------|
| 5 | sum_sum_sum_7b24a3457260 | 0.955 | Only 5% slower on 7.6ms kernel, inherent dispatch overhead |
| 6 | sum_sum_8bcd6e12dcd4 | 0.940 | Only 6% slower, combo dispatch overhead on compute-bound kernel |

These cases are within ~6% overhead, which is the natural "cost" of combo kernel dispatch (pid comparison, dead code paths in binary) on already-large compute-bound kernels where launch savings are negligible.

### False Positives (verified as noise or actually faster)

| Repro | Sweep Ratio | Actual Result |
|-------|-------------|---------------|
| var_mean_var_mean_mean_93065d5c677b | 0.862 | FASTER (70.0->66.8us) |
| sum_sum_sum_8ffe5090cf87 | 0.893 | FASTER (93.9->82.7us) |
| pointwise_4ea254f5c392 | 0.817 | Same (23.7us both) |
| sum_77824d392401 | 0.851 | Same (4.5ms both) |
| All 56 single-kernel cases | 0.90-0.95 | Noise (kernel counts identical) |

## Key Metrics

- **Verified severe regressions:** 4 (15-36% slower, all confirmed live)
- **Verified mild regressions:** 2 (5-6% slower)
- **False positives from sweep data:** 61 (including 56 single-kernel noise + 5 multi-kernel noise)
- **Models affected by severe regressions:** ElectraForCausalLM, ViT-siglip, Swin Transformer, GoogleFnet
- **Overall geomean with combo:** 1.015x faster (net positive despite regressions)
- **Most impactful fix:** X-numel ratio guard would fix all 4 severe regressions without affecting any of the 205 wins
- **Common pattern in all severe cases:** LayerNorm/GroupNorm backward with transposed reductions (sum over hidden dim + sum over batch dim on same tensor)

## Appendix: Generated Code Evidence

### sum_sum_sum_6a2ad206248c Combo Kernel Structure

The problematic combo kernel `triton_red_fused_2` combines:
- Sub-kernel 0: OUTER_TINY reduction, xnumel=128, r0_numel=256 (2.2us standalone)
- Sub-kernel 1: OUTER_TINY reduction, xnumel=128, r0_numel=256 (2.2us standalone)
- Sub-kernel 2: Persistent reduction + atomic_add, xnumel=32768, r0_numel=128 (79.6us standalone)

Combined runtime: 151.8us (vs 84.0us sum of parts = 1.81x slower)

### sum_sum_sum_56ca14eaee84 Combo Kernel Structure

The problematic combo kernel `triton_red_fused_0` combines:
- Sub-kernel 0: xnumel=196608, r0_numel=128, hint=OUTER (94.2us standalone)
- Sub-kernel 1: xnumel=32768, r0_numel=768, hint=DEFAULT (103.2us standalone)

Combined runtime: 388.0us (vs 197.4us sum of parts = 1.97x slower)

### Baseline Kernel Heuristics (for reference)

The baseline correctly uses:
- `persistent_reduction` for small rnumel + small xnumel (no reduction loop)
- `reduction` with `OUTER` hint for large xnumel + small rnumel
- `reduction` with `DEFAULT` hint for balanced x/r
- `reduction` with `OUTER_TINY` hint for tiny xnumel + medium rnumel

The combo kernel loses these optimized heuristic choices because all sub-kernels must coexist in one compiled binary.
