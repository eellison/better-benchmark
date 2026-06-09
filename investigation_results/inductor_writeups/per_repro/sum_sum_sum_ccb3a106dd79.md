# sum_sum_sum_ccb3a106dd79

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_ccb3a106dd79/oracle_multi_output_reduction.py`
- Correctness: PASS
- Oracle: 356.26 us
- Compile (cd=True): 553.92 us
- Ratio: 1.555
- Status: GOOD (significant gap)

## Root Cause

The oracle computes the complete NFNet backward fragment by fusing:
1. avg_pool2d_backward expansion
2. Both sigmoid pointwise chains
3. Both returned channel reductions ([256] outputs)

All in one Triton producer pass with two accumulators plus a tiny partial-sum finalizer. The oracle uses a 2-kernel approach: a partial reduction kernel (C x N_TILES grid) that accumulates per-channel partial sums, followed by a finalize kernel that sums the partials.

Inductor keeps the spatial reduction feeding `sigmoid(arg176) * (1 - sigmoid(arg176))` as a materialized dependent reduction and schedules the sibling channel sum separately. It cannot flatten linear `sum([2,3]) -> broadcast pointwise -> sum([0,2,3])` chains into the same multi-output reduction.

## Kernel Count

- Oracle: 2 kernels (partial reduce + finalize)
- Inductor: Multiple kernels (separate materializations for each dependent reduction)

## Config Exploration

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.555 |
| multi_kernel=2 | 1.584 |
| multi_kernel=3 | 1.555 |

No config helps. The issue is structural: Inductor's reduction codegen cannot reassociate linear dependent reductions into one multi-accumulator kernel.

## File/Line References

- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: cannot fuse sibling reductions that share a common fused producer
- `/tmp/pytorch-work/torch/_inductor/choices.py`: reduction strategy selection does not consider multi-output accumulator patterns
- `/tmp/pytorch-work/torch/_inductor/ir.py`: each reduction is lowered as an independent node

## Design Doc

The fix requires teaching Inductor to reassociate linear dependent reductions and emit one multi-accumulator channel-reduction template over the shared fused producer. Specifically:

1. Recognize pattern: `producer -> sum(dims_a) -> pointwise_a -> sum(dims_b)` alongside `producer -> sum(dims_c)`
2. Flatten into: one kernel iterating over the full spatial domain with multiple accumulators
3. Emit partial-reduction grid (C x tiles) with all accumulators computed per-tile
4. Emit tiny finalize kernel to sum partials

This is the ALGEBRAIC_ELIMINATION pattern: the algebraic simplifier/reduction codegen does not flatten linear sum chains into multi-output reductions. This affects NFNet backward and similar models with shared spatial producers feeding multiple channel-wise reductions.
