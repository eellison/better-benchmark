# Multi-Output Reductions: Optimization Analysis

## Pattern Description

These repros compute **multiple reduction outputs** (2-5 `sum.dim_IntList` calls) from
shared or overlapping input data within a single fused graph fragment. The reductions
share significant pointwise computation (GELU, batch-norm backward, softmax backward)
that must be evaluated before accumulating into their respective outputs.

### Target Repros

| Repro ID | Model | # Sums | Reduction Dims | Key Pattern |
|----------|-------|--------|----------------|-------------|
| sum_sum_sum_70d71fcb0d68 | ConvNeXtV2 (timm) | 5 | [0,2,3] and [2,3] and [1] | GRN/GELU backward with channel-wise and spatial reductions |
| sum_sum_sum_7b24a3457260 | T5-base (torchbench) | 4 | [-1] and [0] | Attention backward: softmax grad + batch sum |
| sum_sum_cdaed89f373c | ResNet18 (torchbench) | 2 | [0,2,3] | BatchNorm backward: sum(grad) and sum(grad * (x-mean)) |
| sum_sum_6a14a9c9ba88 | InceptionV3 (timm) | 2 | [0,2,3] | BatchNorm backward (same pattern, different shape) |

## Current Timing from 3-Config Sweep

All times in microseconds, measured on B200.

| Repro | Default (us) | Combo Persistent (us) | Combo Looped (us) | Best (us) | SOL (us) | Best/SOL | n_kernels |
|-------|-------------|----------------------|-------------------|-----------|----------|----------|-----------|
| 70d71fcb0d68 | 1038 | 994 | 1165 | 940 (coord_descent) | 155 | 6.1x | 7 |
| 7b24a3457260 | 7548 | 7517 | 7245 | 7416 (compiled) | 1676 | 4.4x | 6 |
| cdaed89f373c | 2547 | 2515 | 2514 | 2486 (coord_descent) | 611 | 4.1x | 7 |
| 6a14a9c9ba88 | 1755 | 1626 | 1649 | 1626 (coord_descent) | 243 | 6.7x | 8 |

**Key observation:** These repros are 4-7x above memory-copy SOL. ComboKernel modes
provide marginal benefit (0-4%), suggesting the bottleneck is NOT kernel launch overhead
but rather redundant memory reads within or across the reduction kernels.

## What the Oracle Achieves

Oracle scaffolds exist for `70d71fcb0d68` and `7b24a3457260`. Both specify a
**single-pass multi-accumulator Triton kernel** that:
1. Reads shared input data ONCE
2. Computes shared expressions (GELU, softmax probs, etc.) once
3. Accumulates into multiple output accumulators simultaneously
4. Writes all reduced outputs in the epilogue

The oracle `torch_direct_oracle` for `70d71fcb0d68` demonstrates the algebraic
reformulation that makes single-pass possible, but the actual Triton implementation
is still a TODO scaffold.

## What Inductor Currently Does

### Fusion Behavior

Inductor's scheduler fuses reduction nodes when they share the same `(numel, rnumel)`
pair (see `simd.py:2001-2002`). This means:

1. **Same-dimension reductions DO get fused** into one kernel (e.g., both `sum([0,2,3])`
   calls in `cdaed89f373c` land in the same Triton kernel with 2 separate accumulators).

2. **Different-dimension reductions CANNOT be fused** (e.g., in `70d71fcb0d68`,
   `sum([0,2,3])` vs `sum([2,3])` vs `sum([1])` produce different numels/rnumels and
   become separate kernels).

### The Real Problem: Redundant Input Reads

Even when reductions are fused into one kernel, the shared input tensor is only read
once if the inner_fn expressions are identical. But when different reductions compute
DIFFERENT expressions of the same input (common case in backward passes), the current
codegen structure means:

- Each reduction has its own `inner_fn` that independently loads and computes
- The CSE (Common Sub-Expression Elimination) within a single kernel handles
  literal expression deduplication, so loads of the same buffer at the same index
  are shared
- **However**, the actual bottleneck is that reductions over DIFFERENT dimensions
  require separate kernels, each of which must re-read the large input tensor

For `70d71fcb0d68` (7 kernels, 1028 MB total bytes, 155 us SOL):
- Some kernels reduce over [0,2,3] (numel=320, rnumel=401408)
- Some reduce over [2,3] (numel=40960, rnumel=3136)
- Some reduce over [1] (numel=128, rnumel=320)
- Each kernel reads the 128x320x56x56 input independently

For `cdaed89f373c` and `6a14a9c9ba88` (batch-norm backward):
- The `scatter_add` (max pool backward) dominates and is not fusible with the
  subsequent channel-wise reductions
- The two `sum([0,2,3])` calls ARE fused but the dominant cost is the scatter_add

## What SHOULD Happen (Single-Pass Multi-Accumulator)

### Ideal Implementation

A single Triton kernel should:
1. Tile over the OUTPUT dimensions (e.g., channels C=320)
2. For each output tile, iterate over the full reduction domain (e.g., N*H*W)
3. In the inner loop body, compute ALL shared pointwise expressions once
4. Accumulate into MULTIPLE accumulators (one per output reduction)
5. After the loop, store all reduced outputs

This requires **cross-dimension reduction fusion**: merging reductions that have
DIFFERENT reduction axes but share input data into a single kernel that reads the
data once and routes partial results to the appropriate accumulators.

### Specific Cases

**Case 1: `70d71fcb0d68` (ConvNeXtV2)**
- Output 0: `sum(getitem * gelu * x_n, dims=[0,2,3])` -> shape [320]
- Output 1: `sum(getitem, dims=[0,2,3])` -> shape [320]
- Output 2: `sum(grn_backward * gelu_backward, dims=[0,2,3])` -> shape [320]
- Output 3: `sum(weighted * gelu, dims=[2,3])` -> shape [128, 320, 1, 1]
- Output 4: `sum(neg_spatial_dot * div, dims=[1])` -> shape [128, 1, 1, 1]

Outputs 0-2 share reduction dims [0,2,3] and CAN be fused today.
Output 3 reduces [2,3] and output 4 reduces [1] - these need cross-dimension fusion.

**Case 2: `cdaed89f373c` / `6a14a9c9ba88` (BN backward)**
- Output 0: `sum(grad_output, dims=[0,2,3])` -> shape [C]
- Output 1: `sum(grad_output * (input - mean), dims=[0,2,3])` -> shape [C]

These already share dims and ARE fused. The real opportunity is fusing with the
`scatter_add` that precedes them.

**Case 3: `7b24a3457260` (T5)**
- Two softmax backward blocks with `sum(grad * prob, dim=-1)` followed by
  `sum(result, dim=0)` -- different reduction axes in sequence.
- Cross-dimension fusion could merge these but the graph is complex.

## Which Inductor Files/Functions Need to Change

### 1. Cross-Dimension Reduction Fusion (scheduler)

**File:** `torch/_inductor/codegen/simd.py` (lines 2001-2055)
**File:** `torch/_inductor/scheduler.py` (fusion logic)

Currently, `can_fuse` rejects reduction pairs with different `(numel, rnumel)`.
A new fusion mode would allow merging reductions that:
- Share significant input buffers (e.g., >50% read overlap)
- Have compatible output shapes (one's reduction dim is the other's output dim)

This is similar to `MixOrderReduction` but more general.

### 2. Multi-Accumulator Codegen (triton.py)

**File:** `torch/_inductor/codegen/triton.py` (lines 4451-4530, `reduction()` method)

The kernel already supports multiple `reduction()` calls within a single kernel body.
When reductions share dims, this works. For cross-dimension fusion, the codegen would
need to:
- Generate a loop over the UNION of all reduction dimensions
- Inside the loop, compute shared expressions once
- Accumulate into per-output accumulators with appropriate masking

### 3. Reduction Splitting Heuristics (choices.py)

**File:** `torch/_inductor/choices.py` (lines 508-520)

The stash removes the `numel_hint >= 1024 -> split=1` guard. This is relevant because
split reductions prevent fusion with adjacent pointwise. The multi-output pattern
benefits from NOT splitting.

### 4. Scalar Accumulators (triton.py, codegen)

**File:** `torch/_inductor/codegen/triton.py` (stash additions at line 4430)

Scalar accumulators reduce register pressure by accumulating into [XBLOCK]-sized
scalars instead of [XBLOCK, R0_BLOCK] tiles. This is directly relevant because
multi-output reductions with N accumulators multiply register pressure by N.

### 5. Pattern Recognition (fx_passes)

**File:** `torch/_inductor/fx_passes/post_grad.py`

A new pattern-matching pass could recognize the batch-norm backward template:
```
sum(grad, [0,2,3])     # dgamma component
sum(grad * (x-mu), [0,2,3])  # dbeta component
```
and replace it with a fused `_batch_norm_backward_reduce` op that Triton templates
can lower efficiently.

## Relationship to Stash Scalar Accumulator Work

The stash at `stash@{0}` contains several directly relevant changes:

1. **Scalar accumulators** (`config.triton.scalar_reduction_accumulators`):
   - Reduces register pressure from [XBLOCK, R0_BLOCK] to [XBLOCK] per accumulator
   - Critical for multi-output reductions where N accumulators would otherwise
     require N * R0_BLOCK registers
   - Makes larger R0_BLOCK (2048/4096) viable

2. **Persistent reduction threshold raised** (INNER: 1024 -> 2048):
   - Keeps more reductions in persistent mode, which is natural for multi-output
     since all accumulators can use the same persistent tile

3. **Split-reduction guard removed** (numel >= 1024 no longer forces split=1):
   - Allows inner reductions with many output elements to split, enabling
     better parallelism for channel-wise reductions (numel=320, rnumel=401408)

4. **R0_BLOCK expansion configs** (2048, 4096, 8192, 16384):
   - Paired with scalar accumulators, fewer iterations means less loop overhead
     and better instruction-level parallelism for multi-accumulator kernels

5. **Online softmax rnumel threshold**:
   - Not directly relevant to sum-only reductions but shows the pattern of
     adapting codegen strategy based on reduction size

## Recommended Next Steps

1. **Immediate (stash-based):** Apply scalar accumulator + R0_BLOCK expansion from
   stash. Measure impact on these repros with coordinate_descent_tuning. Expected
   benefit: 10-30% from reduced iteration count and better register utilization.

2. **Short-term (pattern recognition):** Add batch-norm backward reduction template
   that fuses `sum(grad)` + `sum(grad*(x-mu))` into one optimized kernel. This
   directly benefits `cdaed89f373c` and `6a14a9c9ba88`.

3. **Medium-term (cross-dimension fusion):** Implement a new scheduler fusion mode
   that can merge reductions over different dimensions when they share inputs.
   This benefits `70d71fcb0d68` most. Requires:
   - Input overlap analysis in the scheduler
   - New kernel codegen mode that iterates over the union of reduction dims
   - Careful handling of output indexing

4. **Measurement priority:** The batch-norm backward cases (`cdaed89f373c`,
   `6a14a9c9ba88`) are dominated by `scatter_add` (max pool backward), so
   multi-output reduction fusion alone won't close the SOL gap. Focus measurement
   on `70d71fcb0d68` where the 7 kernels are all pure reductions.
