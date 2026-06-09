# any_amax_sum_4a963b47b28d


## Measured Timings
- Oracle: 25.73 us
- Compile (CDT): 28.13 us
- Ratio: 1.09x

## Classification: CONSTANT_FOLDING (partially fixed)

## Pattern

M2M100 attention softmax with tautological mask: [1024, 128, 128] scores
viewed as [64, 16, 128, 128], generated iota/add/unsqueeze/ge/expand mask
where iota >= 0 is always True, where selects 0 bias (identity add),
any(eq(-inf)) guard (always False), stable softmax, view back to [1024, 128, 128].

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 25.73 | 1.000 |
| torch.compile (before fix) | ~28.74 | 1.098 |
| torch.compile (after fix) | ~28.1 | 1.092 |

Correctness: PASS (max_diff=2.98e-08)

## Fix Applied

Commit 3e627768c9f in /tmp/pytorch-work:
- Extended `UniformValueConstantFolder._deduce_value` to propagate uniform tensors
  (all elements equal, numel>1) through view/expand ops by reducing to size-1
- Added `run_node` override to handle expand with unknown shape args on uniform inputs
- This folds: `iota(128) >= 0` -> `full(True, [1,1,128,1])` -> `expand` -> `where(_, 0.0, -inf)` -> `full(0.0)` -> `add(input, 0.0)` eliminated by remove_no_ops

## Kernel Count

- Oracle: 1 kernel (pure row-softmax with exp2, no mask computation)
- Inductor: 1 kernel (fused softmax, dead mask+add eliminated, but still has eq(-inf)+any guard)

## Remaining Gap (1.092x)

The oracle uses a hand-tuned row-softmax kernel with:
- exp2 with fused log2(e) scaling instead of exp
- BLOCK_ROWS multi-row tiling
- Combined row_has_value check with softmax (no separate any reduction)
- No separate eq(-inf)/any_dim guard for all-masked rows

Inductor's decomposed softmax still computes eq(-inf) + logical_not + any.dim as
a separate reduction before the main amax/exp/sum/div chain. This is semantically
necessary (can't statically prove input has no -inf values) but the oracle proves
it unnecessary by construction.

## File References

- Fix: `/tmp/pytorch-work/torch/_inductor/fx_passes/joint_graph.py` (UniformValueConstantFolder)
- Pattern: `_deduce_value` view op handler and `run_node` override

## Status: PARTIALLY_FIXED (1.098x -> 1.092x)
Residual gap is ONLINE_SOFTMAX: oracle fuses the all-masked-row guard into the
softmax computation; Inductor's decomposed pattern keeps it as separate reduction.
