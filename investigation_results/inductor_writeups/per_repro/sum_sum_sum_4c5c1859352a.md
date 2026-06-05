# sum_sum_sum_4c5c1859352a

## Compile: 14.43us, Oracle: 9.86us, Gap: 1.464x

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Root Cause

The oracle computes the complete 3-output graph (functorch dp_cifar10 BN-backward-style) by streaming the added-and-masked producer once per (batch, two-channel group), keeping `sum(where)` and `sum(where * arg52_1)` over each 8x8 spatial block in registers, and deriving the dependent grouped reductions plus full tensor epilogue from those shared summaries.

Inductor generates 4 kernels:
1. `triton_per_fused_add_le_mul_sum_view_where_0`: persistent reduction for spatial sums
2. `triton_poi_fused_1`: pointwise intermediate
3. `triton_per_fused_2`: second reduction pass
4. `triton_poi_fused_add_le_mul_neg_sub_sum_unsqueeze_view_where_3`: final pointwise epilogue

The fundamental issue is an ALGEBRAIC_ELIMINATION opportunity: the oracle recognizes that two base spatial summaries (sum(masked_input) and sum(masked_input * weight)) drive all downstream computation. Inductor's simplifier does not propagate this factoring through the view-threaded dependent reduction chain, so it materializes and re-reads intermediates multiple times.

## Kernel Count
- Oracle: 1 kernel (shared spatial sums + dependent epilogue)
- Inductor: 4 kernels (separate reductions + pointwise + epilogue)

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo_kernels + CDT | 14.43 | baseline (1.464x) |
| multi_kernel=2/3 | unlikely to help (algebraic pattern) |

## Fix Assessment: Design doc

This requires an algebraic rewrite pass that recognizes the grouped BN-backward-style pattern:
1. Two spatial reductions share the same masked producer
2. Downstream channel reductions and tensor epilogue depend only on those two base sums
3. Everything can be fused into one kernel that computes spatial sums in registers and derives all outputs

### What's needed:
A guarded rewrite (FX pass or scheduler pattern) that identifies when:
- Multiple sibling reductions share a common producer
- Dependent operations use only those reduction results
- The full tensor epilogue can be computed from the shared sums

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`: pattern matching for shared reduction factoring
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: multi-output reduction template

### Affected repro count:
This pattern (BN-backward-style shared spatial reductions) affects several functorch dp repros. Likely 3-5 in the corpus.

## Details
- Model: torchbench_functorch_dp_cifar10 (train)
- Shape: [64, 64, 8, 8] f32 (multiple inputs)
- Pattern: add -> where(le, 0, x) -> mul(weight) -> sum(spatial) x2 -> dependent channel ops -> full tensor epilogue
- Outputs: [64, 64, 8, 8] f32 tensor + [64] f32 vector + [64] f32 vector
