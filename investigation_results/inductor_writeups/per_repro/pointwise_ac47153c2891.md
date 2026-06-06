# pointwise_ac47153c2891

## Classification: SCATTER_PERMUTATION_INVERSION

## Current Result

- Oracle path: `repros/canonical/pointwise_ac47153c2891/oracle_layout_index_fusion.py`
- Correctness: PASS
- Oracle: 38.56 us
- Compile (cd=True): 47.07 us
- Ratio: 1.221
- Status: GOOD

## Root Cause

The oracle computes the inverse-permutation layout materialization by scattering each bmm_2.view(8, 12, 4096, 64)[b, h, q, :] vector directly to its final contiguous [b * 4096 + getitem_3[b, h, q], h * 64:(h + 1) * 64] output slice in a single kernel.

Inductor materializes the scatter(iota) inverse-index tensor, gathers through that tensor, then permutes/clones/views the result. This requires:
1. Building the inverse permutation index tensor via scatter
2. Gathering through the index tensor
3. Permuting and cloning to final layout

The oracle avoids the intermediate index materialization by directly scattering to the destination layout.

## Kernel Count

- Oracle: 1 kernel (direct destination-layout scatter)
- Inductor: 3+ kernels (scatter index build, gather, permute/clone)

## Config Exploration

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.221 |
| multi_kernel=2 | 1.241 |
| multi_kernel=3 | 1.240 |

No config helps. multi_kernel slightly worsens. The issue is structural: Inductor does not recognize scatter-built inverse permutations as a pattern.

## File/Line References

- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: treats scatter index construction and downstream gather as separate memory effects
- `/tmp/pytorch-work/torch/_inductor/fx_passes/`: no pass to rewrite scatter-built inverse permutations

## Design Doc

The fix requires a layout-indexing fusion pass that rewrites scatter-built inverse permutations feeding gather plus clone into a direct destination-layout scatter over the original permutation. This eliminates the intermediate index tensor materialization and the extra gather kernel.

Pattern: `scatter(zeros, perm, iota) -> gather(data, inverse_perm) -> clone` should become `scatter(output, perm, data)`.

This is a data-dependent indexing pattern common in attention heads where query positions are reordered (e.g., by a sorting/bucketing step).
