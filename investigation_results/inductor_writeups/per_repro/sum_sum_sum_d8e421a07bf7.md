# sum_sum_sum_d8e421a07bf7

## Classification: ALGEBRAIC_ELIMINATION

## Benchmark Results
- Oracle: 23.46 us
- Inductor compiled: 43.9 us
- Ratio: 1.872x (oracle is 1.87x faster)
- With multi_kernel=3: 39.64 us (still slower than oracle)

## Kernel Count
- Inductor: 1 kernel (fused but suboptimal strategy)
- Oracle: 2 kernels (summary reduce + final epilogue)

## Root Cause

The oracle computes the complete DenseNet batch-norm-backward tail by reading the masked producer, centered input, residual slice, and affine vectors once, accumulating `sum(where)`, `sum(where * centered)`, `sum(centered)`, and `sum(slice)` per channel in a single multi-accumulator reduction pass, then deriving both returned `[32]` vectors from these summaries in a small epilogue.

The repro has this structure:
1. `sum(where_self, [0,2,3])` -> [32] channel reduction
2. `sum(where_self * centered, [0,2,3])` -> [32] channel reduction
3. BN-backward arithmetic using those two sums to produce a [128,32,32,32] intermediate
4. `sum(intermediate + slice, [0,2,3])` -> final [32] channel reduction

The key insight is that the final sum (step 4) can be algebraically decomposed: `sum((where - mean_broadcast - centered*coef_broadcast) * scale_broadcast + slice)` simplifies to a linear combination of `sum(where)`, `sum(centered)`, `sum(slice)` and constants. The oracle exploits this identity to compute all four partial sums in one pass over the data.

Inductor preserves the literal dependent-reduction DAG: it computes the first two channel reductions, materializes the BN-backward arithmetic as a full [128,32,32,32] intermediate, adds the slice, and then performs the final channel reduction. Even though it fuses into 1 kernel, the single kernel must make multiple passes or has suboptimal memory access patterns for the dependent reduction.

## Config Exploration
- `combo_kernels=True, coordinate_descent_tuning=True`: 43.9 us (baseline)
- `+ triton.multi_kernel=3`: 39.64 us (slight improvement, still 1.69x gap)

## What Inductor Needs (Design Doc)

**Enhancement needed**: Algebraic dependent-reduction rewrite pass in `torch/_inductor/fx_passes/`.

The pass would:
1. Detect the BN-backward pattern: two sibling channel reductions followed by broadcast arithmetic whose result is reduced again over the same axes
2. Prove the linear identity that the final dependent reduction can be expressed as a linear combination of additional sibling summaries (sum(centered), sum(slice))
3. Rewrite the graph to lower all summaries as one multi-accumulator reduction kernel + a small vector epilogue

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` - register the new pattern
- `/tmp/pytorch-work/torch/_inductor/fx_passes/` - new file for the BN-backward algebraic rewrite

**Affected repro count**: This pattern appears across DenseNet, ResNet, and other BN-backward scopes. The `sum_sum_sum` family likely has 10+ repros with this same algebraic elimination opportunity.
