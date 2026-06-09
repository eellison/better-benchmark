# sum_sum_sum_041aaca79573

## Summary

- Model: Swin Transformer (relative position bias backward)
- Oracle: `oracle_relative_position_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 2.03x (oracle 270.4us, compile 548.7us)
- Kernel count: Inductor 3 kernels, Oracle fewer (fused scatter-reduce)

## Root Cause

The repro computes the Swin Transformer relative-position-bias backward. It reduces a batch-summed `[8192, 4, 49, 49]` producer directly into duplicate `[169, 4]` relative-position buckets using `index_put(accumulate=True)`, while simultaneously writing a required `[32768, 49, 49]` softmax-backward side output.

The oracle fuses the batch reduction with the indexed bucket accumulation:
1. Computes `sum(dim=0)` over the batch dimension
2. Applies softmax backward (mul -> sum -> fma)
3. Accumulates results into [169, 4] buckets via structured scatter-add
4. All done in fewer kernel launches by recognizing the scatter pattern

Inductor separates these into:
1. Batch reduction kernel (sum over dim=0)
2. Pointwise softmax-backward epilogue
3. Separate index_put(accumulate=True) scatter kernels for the two [169, 4] outputs

The 2.03x gap comes from materializing large intermediates ([8192, 4, 49, 49] -> [4, 49, 49]) and running the scatter as a separate pass.

## Config Exploration

| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT (default) | 548.7 |
| multi_kernel=2 | 472.4 |
| multi_kernel=3 | 473.1 |

multi_kernel=2/3 provide ~14% improvement but don't close the 2x gap. The structural issue (separate scatter kernels) is the dominant cost.

## Fix Assessment

**Design doc** -- requires a structured scatter-reduce lowering.

### What's needed:
A new FX pass or scheduler enhancement that recognizes the pattern:
`sum(dim=0) -> permute -> view -> index_put(accumulate=True, indices=[relative_position_index])`

This is a structured scatter-reduce where:
- The reduction domain is the batch dimension
- The scatter indices map 49*49 positions to 169 relative-position buckets
- The scatter can be fused into the reduction (accumulate partial sums per bucket during the batch reduction)

### Difficulty: High
The index_put with accumulate=True using non-trivial index tensors (relative position table) makes this hard to fuse generically. The scatter indices are data-dependent (though constant at runtime).

## Relevant Files

- `/tmp/pytorch-work/torch/_inductor/lowering.py`: index_put lowering with accumulate
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: fusion of reduction + scatter epilogue
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`: potential pattern-matching pass

## Affected Repro Count

This Swin relative-position-bias backward pattern likely affects multiple Swin variants. The SCATTER_REDUCE classification covers many repros in this corpus.
