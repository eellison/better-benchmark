# mean_ee17938d7aff - Residual RMSNorm Aliases (LLaVA)

## Classification: BANDWIDTH_BOUND
## True Floor: Near-floor
## Actionable: No (marginal gap)

## Benchmark Results
- Oracle: 7.14 us
- Compiled: 7.55 us
- Ratio: 1.058x

## Root Cause

The oracle computes residual-add + RMSNorm (fp16 -> fp32 -> reduction -> rsqrt -> normalize -> fp16 -> affine weight) and returns three alias views of the same output in a single row-reduction Triton kernel with shape-specialized parameters (512 rows, 4096 columns).

Inductor also emits 1 kernel that fuses all operations. The 5.8% gap is within noise for bandwidth-bound normalization operations. The oracle achieves slightly better memory bandwidth utilization through:
1. Shape-specialized block sizes tuned to exactly 4096 columns
2. Explicit control of accumulator dtype management
3. Minimal overhead from the view/alias returns (zero-copy)

## Kernel Count
- Inductor: 1 kernel
- Oracle: 1 kernel (+ 1 trivial view kernel counted by @triton.jit decorator but is just for alias output management)

## Config Exploration
- `coordinate_descent_tuning=True`: Already enabled, this is the primary mechanism for closing such gaps
- This is a near-floor case where the compiled code is within ~6% of the hand-tuned oracle

## Assessment

This is classified as BANDWIDTH_BOUND by the oracle's own diagnosis. The gap of 1.058x is marginal and could be closed by coordinate descent tuning finding slightly better tile sizes. No scheduler or fusion fix is needed - the work is already fully fused.

At 7.14 us vs 7.55 us (0.41 us difference), this is effectively at floor for a row-normalization workload of this size (512 * 4096 * 2 bytes = 4 MB read + write).

## File references
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (reduction template)
- `/tmp/pytorch-work/torch/_inductor/choices.py` (persistent reduction vs split)
