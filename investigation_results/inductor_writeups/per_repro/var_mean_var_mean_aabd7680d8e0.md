# var_mean_var_mean_aabd7680d8e0 -- Chained BN Recompute-Fusion


## Measured Timings
- Oracle: 22.24 us
- Compile (CDT): 21.31 us
- Ratio: 0.96x

**Classification:** RECOMPUTE_FUSION  
**Gap:** 1.55-1.77x (oracle ~22 us, Inductor ~36 us)  
**Source model:** timm_visformer_small (training)  
**Pattern:** Two chained BatchNorm reductions over the same spatial dims  
**Kernel count:** Inductor 1 kernel (3 internal loops) vs Oracle 2 kernels  

## 1. Computation Structure

```
Input: x[128, 768, 7, 7]  (NCHW, strides=(37632,49,7,1), 18.4 MB)
Residual: res[1, 768, 7, 7] (150 KB)
Per-channel params: weight1, bias1, weight2, bias2, running stats (each [768])

BN1:
  var1, mean1 = var_mean(x, dim=[N,H,W])       # Reduction: 6272 elements/channel, 768 channels
  y1 = (x - mean1) * rsqrt(var1+eps) * weight1 + bias1 + residual

BN2:
  var2, mean2 = var_mean(y1, dim=[N,H,W])      # Reduction: 6272 elements/channel, 768 channels
  output = (y1 - mean2) * rsqrt(var2+eps) * weight2 + bias2

+ Running stat EMA updates (cheap, per-channel only)
```

Key structural dependency: y1 is consumed by BOTH:
- A reduction (var_mean_2 computes stats over y1)
- A pointwise (BN2 normalize uses y1 element-wise after the reduction)

This "reduce-then-use" pattern is the fundamental root cause.

## 2. Inductor's Generated Kernel (Non-Persistent, Default)

Relevant codegen path: `/tmp/pytorch-work/torch/_inductor/codegen/simd.py` line 2155 (`generate_node_schedule`)

Inductor fuses everything into 1 kernel with **3 sequential reduction loops**:

```
Loop 1: Welford reduce(x) -> mean1, var1
         Reads x from global (18.4 MB)

Loop 2: Recompute BN1: y1 = normalize(x, mean1, var1) * w1 + b1 + res
         Welford reduce(y1) -> mean2, var2
         STORES y1 to in_out_ptr0 (18.4 MB write)  <<< KEY INEFFICIENCY
         Reads x again (18.4 MB) + residual (150 KB)

Loop 3: Loads y1 from in_out_ptr0 (18.4 MB read)   <<< KEY INEFFICIENCY
         output = (y1 - mean2) * invstd2 * w2 + b2
         Stores output to in_out_ptr0 (18.4 MB write)
```

**Total memory traffic: ~92 MB** (3x read of 18.4 MB data + 2x write of 18.4 MB)

### Why 3 loops?

The scheduler in `simd.py:generate_node_schedule()` uses `requires_closing_previous_reduction()` (line 2215):
- After reduction 1 produces mean1/var1, the y1 computation DEPENDS on them, closing loop 1
- The y1 computation + reduction 2 are in loop 2 (y1 is the input to reduction 2)
- But y1 must be STORED to global because the output pointwise (loop 3) needs it AFTER reduction 2 completes
- The output pointwise depends on mean2/var2 (not yet available during loop 2's iteration)
- This forces a buffer materialization of y1 between loops 2 and 3

The fundamental limitation: **Inductor's non-persistent codegen model cannot keep a value in registers across a reduction boundary.**

### Generated kernel details
```
xnumel = 768 (channels), r0_numel = 6272 (N*H*W per channel)
Grid: ceil(768/XBLOCK) blocks
Signature: 17 pointer args + 2 size args + XBLOCK/R0_BLOCK constexpr
num_reduction = 4 (Welford mean+m2 for each of 2 var_means)
```

## 3. Oracle Strategy

The oracle splits into 2 kernels, each with 768 thread blocks (1 per channel), using
BLOCK_K=8192 (>= 6272=elements_per_channel) for **persistent single-tile reduction**:

**Kernel 1** (`_first_bn_stats_kernel`): Compute BN1 statistics only
- Reads x (18.4 MB), writes mean1/invstd1/running_mean1/running_var1 (~12 KB)
- Uses sum/sum_sq formula: mean = sum(x)/N, var = sum(x*x)/N - mean^2
- Simple structure: load channel data, reduce, store scalars

**Kernel 2** (`_second_bn_recompute_output_kernel`): RECOMPUTES y1 + computes BN2 + stores output
- Reads x (18.4 MB) + residual (150 KB) + mean1/invstd1/w1/b1 from kernel 1 (~12 KB)
- BLOCK_K=8192 >= 6272 -> **entire channel fits in one tile (persistent)**
- Single-pass execution (all in registers):
  1. Load x for channel into register tile (6272 f32 values = 32 per thread at 256 threads)
  2. Load residual (49 values, broadcast over N dimension)
  3. RECOMPUTE y1 = (x - mean1) * invstd1 * w1 + b1 + residual  (ALU only, no global write)
  4. Reduce y1 -> sum, sum_sq -> mean2, var2, invstd2  (**y1 still in registers!**)
  5. Compute output = (y1 - mean2) * invstd2 * w2 + b2  (**y1 still in registers!**)
  6. Store output to global memory (18.4 MB)

**Total memory traffic: ~55 MB** (2x read 18.4 MB + 1x write 18.4 MB + negligible scalars)

### Why recomputation wins over fusion

The oracle PAYS: 18.4 MB extra READ (re-reads x in kernel 2)
The oracle AVOIDS:
- 18.4 MB WRITE of y1 (never materializes y1)
- 18.4 MB READ of y1 (no loop 3 reload)

**Net savings: 18.4 MB** (one full tensor round-trip).
At ~3 TB/s HBM bandwidth (B200): 18.4 MB / 3 TB/s = **~6 us**, matching the observed gap.

## 4. Persistent Variant Analysis (multi_kernel=3)

With `config.triton.multi_kernel=3`, Inductor generates BOTH a non-persistent (`triton_red_`) and a persistent (`triton_per_`) variant, benchmarks them, and picks the winner.

The persistent variant (`triton_per_fused_add_copy__mul_rsqrt_squeeze_sub_unsqueeze_var_mean_1`):

```python
R0_BLOCK: tl.constexpr = 8192   # Entire channel in one tile (>= 6272)
XBLOCK: tl.constexpr = 1        # One channel per thread block

# All loads happen ONCE at the start:
tmp0 = tl.load(x_ptr + ...)         # x[channel, :] - 6272 elements in registers
tmp40 = tl.load(residual_ptr + ...)  # residual[channel, :] - 49 elements

# First reduction (sum-based, not Welford - more efficient for persistent):
tmp7 = tl.sum(x, 1)                 # sum of x
mean1 = tmp7 / 6272
tmp12 = (x - mean1) * (x - mean1)   # squared deviations
tmp16 = tl.sum(tmp12, 1)            # sum of squared deviations
var1 = tmp16 / 6272

# BN1 normalize + affine (y1 stays in registers as tmp41):
invstd1 = rsqrt(var1 + eps)
y1 = (x - mean1) * invstd1 * w1 + b1 + residual   # tmp41

# Second reduction (y1 in registers, no reload!):
tmp48 = tl.sum(y1, 1)               # sum of y1
mean2 = tmp48 / 6272
tmp55 = tl.sum((y1-mean2)^2, 1)     # sum of squared deviations
var2 = tmp55 / 6272

# BN2 normalize (y1 still in registers!):
output = (y1 - mean2) * rsqrt(var2+eps) * w2 + b2   # tmp73

# Stores:
tl.store(buf, y1)       # LINE 504: *** DEAD STORE *** (overwritten immediately!)
tl.store(buf, output)   # LINE 508: actual output (same address as line 504)
```

**Key finding: The persistent kernel has the CORRECT algorithm (keeps y1 in registers), but emits a DEAD STORE of y1.**

### Measured performance with multi_kernel=3: **~34 us** (vs 36 us default, 22 us oracle)

## 5. The Dead Store Problem

In the persistent kernel, lines 504 and 508 both store to `in_out_ptr0 + (r0_1 + 49*x0 + 37632*r0_2)` -- the exact same address. Line 504 stores y1 (tmp41), line 508 stores output (tmp73).

**This dead store exists because:**
- The scheduler allocates `buf8` as a shared buffer for both y1 and the final output
- In the NON-persistent variant (3 loops), the store of y1 in loop 2 is NECESSARY (loop 3 reads it back)
- When generating the persistent variant, the same store instruction is preserved even though the persistent kernel keeps y1 in registers and never reloads it
- The codegen lacks a dead-store elimination pass for persistent reductions

**Cost of dead store:** 768 * 6272 * 4 bytes = 18.4 MB extra write. At 3 TB/s = ~6 us waste.

## 6. Memory Traffic Comparison

| Strategy | Reads | Writes | Total I/O | Measured |
|----------|-------|--------|-----------|----------|
| Inductor (3 loops, default) | 55.3 MB | 36.8 MB | 92.0 MB | 36 us |
| Oracle (2 kernels, recompute) | 36.9 MB | 18.4 MB | 55.3 MB | 22 us |
| Inductor persistent (actual, w/ dead store) | 18.6 MB | 36.8 MB | 55.4 MB | 34 us |
| Inductor persistent (ideal, no dead store) | 18.6 MB | 18.4 MB | 37.0 MB | ~18 us (theoretical) |

The ideal persistent kernel would have LESS I/O than oracle (37 MB vs 55 MB) because it reads x only ONCE.

### Bandwidth utilization analysis

At B200 HBM3e bandwidth of ~3 TB/s:
- Oracle achieves: 55.3 MB / 22 us = 2.5 TB/s (83% utilization)
- Inductor default achieves: 92.0 MB / 36 us = 2.6 TB/s (85% utilization)
- Inductor persistent: 55.4 MB / 34 us = 1.6 TB/s (54% utilization) -- low occupancy!

The persistent kernel has poor bandwidth utilization, suggesting register pressure limits occupancy.

## 7. Register Pressure / Occupancy Analysis

**B200 specs:** 148 SMs, 65536 regs/SM, 2048 max threads/SM, 1024 max threads/block

For the persistent kernel with XBLOCK=1, R0_BLOCK=8192, 8 warps (256 threads):
- Elements per thread: 8192 / 256 = 32
- Register estimate per thread:
  - x tile: 32 regs (kept for deviation computation)
  - deviation values: 32 regs (x - mean1, reusable)
  - y1 tile: 32 regs (must keep for post-reduction use)
  - Welford/sum accumulators: 4-6 regs
  - Per-channel scalars: 8-10 regs
  - Index computation: 5 regs
  - **Total estimate: 80-110 regs/thread**
- Blocks per SM: floor(65536 / (100 * 256)) = **2 blocks/SM**
- Active threads per SM: 2 * 256 = 512 (25% occupancy)
- Total waves: ceil(768 / (148 * 2)) = **3 waves**

Compare to oracle kernel 2:
- Same persistent approach but potentially fewer live values
- Oracle doesn't need x after computing y1 (can reuse registers)
- Estimated: 50-70 regs/thread -> 3-5 blocks/SM -> better occupancy

## 8. Config Exploration Results

| Config | compile_us | oracle_us | ratio |
|--------|-----------|-----------|-------|
| Default (coordinate_descent + combo_kernels) | 36.38 | 20.51 | 1.774x |
| + multi_kernel=3 | 34.08 | 22.59 | 1.508x |
| + multi_kernel=3 (run 2) | 33.89 | 23.49 | 1.443x |
| + persistent_reductions=True (no effect, threshold blocks) | 36.45 | 21.54 | 1.692x |
| + cooperative_reductions (from previous writeup) | 29.40 | - | - |
| **+ persistent_reduction_threshold_inner=8192 (FIX)** | **23.46** | **23.49** | **0.999x** |

**Fix implemented:** `persistent_reduction_threshold_inner=8192` closes the gap entirely (at floor).

## 9. Proposed Fixes (Prioritized)

### Fix 1: Raise persistent reduction threshold for chained reductions (IMPLEMENTED)

**File:** `/tmp/pytorch-work/torch/_inductor/choices.py` line 430
**Commit:** `d2cd52d3117` on branch `pr-184905`

The threshold was 1024 for INNER reductions. With rnumel=6272, the persistent variant was not generated by default (only with multi_kernel enabled).

**Change:** New config `config.triton.persistent_reduction_threshold_inner = 8192` (env: `TORCHINDUCTOR_PERSISTENT_REDUCTION_THRESHOLD_INNER`). The threshold dict now reads the config instead of hardcoding 1024.

**Result on this repro:** 1.55x -> 1.00x (at floor, compile matches oracle)

**Rationale:** With 8 warps (256 threads), RBLOCK=8192 means 32 elements/thread. Even with BN-level math (x, deviations, y1, accumulators), this stays under 128 regs/thread, giving at least 2 blocks/SM.

### Fix 2: Dead store elimination in persistent reduction codegen (HIGH PRIORITY)

**File:** `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` or `/tmp/pytorch-work/torch/_inductor/codegen/simd.py`

When generating a persistent reduction kernel body, detect and eliminate stores where:
1. The same output address is stored to more than once within the kernel
2. There is no intervening load from that address between the two stores
3. The earlier store value is not used elsewhere

**Implementation approach:**
After `codegen_body()` generates the kernel code, scan `self.stores` and `self.post_loop_store` for duplicate store targets. If a buffer location is stored twice with no load between, eliminate the first store.

**Expected improvement:** Eliminates 18.4 MB write -> saves ~6 us -> brings gap from 1.5x to ~1.2x.

### Fix 3: Scheduler enhancement for "reduce-then-use" pattern (MEDIUM PRIORITY, DESIGN)

**File:** `/tmp/pytorch-work/torch/_inductor/codegen/simd.py` line 2215 (`requires_closing_previous_reduction`)

When the scheduler detects:
```
y = f(x)              # pointwise producer
stats = reduce(y)     # reduction consumer
output = g(y, stats)  # pointwise consumer needing BOTH y and reduction output
```

And the kernel will be persistent (all elements fit in one tile), the scheduler should NOT materialize y between the reduction and post-reduction use. Instead:
- Keep y as a "virtual register" variable in the node schedule
- Emit both the reduction and the post-reduction pointwise in the same code section
- The persistent code already achieves this structurally (lines 476-501 in the persistent kernel), but the dead store (line 504) reveals the scheduler still thinks materialization is needed

This fix is mostly about telling the scheduler "in persistent mode, the intermediate doesn't need realization" and removing the store from the codegen.

### Fix 4: Welford vs sum/sum_sq optimization (LOW PRIORITY)

The non-persistent kernel uses Welford online algorithm (necessary for numerical stability with tiled reduction). The persistent kernel already uses sum-based approach. No fix needed for persistent path.

## 10. Affected Repro Count

**Directly affected** (same chained var_mean pattern with recompute opportunity):
- `var_mean_var_mean_aabd7680d8e0` (this repro, 1.77x)
- `var_mean_var_mean_6d7a29cb97f1` (RECOMPUTE_FUSION, DenseNet stem)
- Other `var_mean_var_mean_*` repros: 10+ in corpus (potentially all 15+)

**Broader pattern** (reduce-then-use with cheap recomputable intermediate):
- `mean_8b70c08405c1` (RECOMPUTE_FUSION, residual RMSNorm)
- `mean_d5150760404d` (RECOMPUTE_FUSION, residual RMSNorm)
- `mean_var_mean_*` repros (chained LayerNorm + BN patterns)
- Any repro where a reduction's input is also needed after the reduction

**Estimated total:** 15-25 repros would benefit from Fix 1 + Fix 2.

## 11. General Design Principle

**The "reduce-then-use" pattern** is a general class of optimization challenges:

```
intermediate = expensive_or_cheap_compute(inputs)
stats = reduce(intermediate)          # needs ALL elements
output = transform(intermediate, stats)  # needs EACH element + stats
```

Three strategies exist:
1. **Materialize** (Inductor default): Store intermediate, run reduce, reload for transform. Cost: 2x intermediate I/O.
2. **Recompute + split** (Oracle): Compute stats in kernel 1, recompute intermediate in kernel 2 for transform. Cost: 2x input read.
3. **Persistent fusion** (Inductor persistent, ideal): Keep intermediate in registers, do reduce AND transform in one pass. Cost: 1x input read + 1x output write.

Strategy 3 is optimal when it's feasible (elements fit in registers). Strategy 2 beats strategy 1 when intermediate compute is cheap and the intermediate tensor is large relative to inputs.

Inductor currently defaults to strategy 1 (materialize) and only uses strategy 3 when multi_kernel is enabled AND the persistent threshold allows it. It never uses strategy 2 (recompute + split) because the scheduler doesn't have a "recompute producer for dependent consumer" code path.

**The fix priority is: make strategy 3 (persistent fusion) work by default for this size range, and eliminate the dead store that degrades it.**
