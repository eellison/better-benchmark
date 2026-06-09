# pointwise_083e5f8c7cdc - Reformer Inverse Gather

## Benchmark Result
- Oracle: 10.14 us
- Compile: 19.33 us
- Ratio: 1.905x
- Status: GOOD (significant gap)

## Root Cause
The oracle computes the Reformer inverse-permutation gather in 2 specialized kernels:
1. `_invert_permutation_kernel`: inverts the per-head getitem_29 permutation (12 heads x 4096 seq)
2. `_gather_final_layout_kernel`: gathers bmm_14 rows directly into the final contiguous [4096, 768] token-major layout

Inductor lowers the same operation as:
1. `triton_poi_fused_0`: scatter + iota generation (builds inverse permutation via scatter, materializes intermediates)
2. `triton_poi_fused_clone_expand_gather_permute_unsqueeze_view_1`: generic gather + permute + clone into final layout

The fundamental issue is that Inductor's generic scatter/gather lowering materializes intermediate buffers (the expanded iota, the scatter output, the gather output before permute/clone) that the oracle avoids by directly computing the inverse permutation and writing to the final layout in one step.

## Kernel Count
- Oracle: 2 kernels (invert permutation, gather to final layout)
- Inductor: 2 kernels (scatter+iota, gather+permute+clone+view)

## Config Exploration
- `combo_kernels = True`: no effect (these are pointwise, not reductions)
- `coordinate_descent_tuning = True`: may help tile sizes but doesn't fix the fundamental materialization issue

## File/Line References
- `/tmp/pytorch-work/torch/_inductor/lowering.py`: scatter and gather lowered as generic operations
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: cannot recognize the scatter-then-gather pattern as one inverse-permutation-gather

## Design Doc
This is a NEW_PATTERN case. Inductor's scheduler/codegen pattern library does not recognize the Reformer idiom where:
1. A permutation is inverted via scatter(iota)
2. The inverse is used to gather into a final contiguous layout with head-to-hidden reshaping

The fix would be an FX pass in `torch/_inductor/fx_passes/` that detects:
- `empty -> iota -> view -> expand -> scatter(src=expanded_iota, index=perm)` followed by
- `unsqueeze -> expand -> gather(src=data, index=inverse_perm) -> permute -> clone -> view`

and replaces it with a single fused "inverse_permutation_gather" node that directly computes the output. This is inherently a pattern recognition problem, not a scheduler fusion problem, because the intermediate scatter output has a data dependency that prevents simple fusion.

### Affected Repro Count
This pattern is specific to Reformer-style models that use reversible attention with per-head permutations. Likely affects 2-4 repros in the corpus.
