# sum_sum_ed9e2feedf2a

## Compile: 33.6us, Oracle: 30.59us, Gap: 1.098x

## Diagnosis: MULTI_OUTPUT_SHARED_REDUCTION

## Root cause

The oracle computes the full DenseNet-121 BN-backward tail (C=384) by:
1. A cooperative partial-reduce kernel that streams the masked `where` producer across all N*H*W spatial elements, computing dual partial sums (sum_where and sum_where*centered) per channel tile
2. A finalization kernel that reduces partials and computes the returned `sum_centered * invstd` vector
3. A fused epilogue kernel that adds all 20 residual slices (from channels 352:384) and computes the BN-backward input gradient using the finalized scalars

Inductor emits 2 kernels:
1. A persistent reduction kernel computing the dual where+sum reductions over all C=384 channels
2. A pointwise epilogue computing the 20-way residual slice-add + grad

The 1.098x gap comes from the oracle's ability to share the masked/centered producer computation between the two sibling reductions AND fuse the dependent epilogue using finalized reduction results, all while keeping per-channel tiling. Inductor's scheduler materializes the intermediate reduction results before the epilogue can consume them, requiring an extra global memory round-trip for the per-channel scalars.

## Config exploration

| Config | Compile (us) | Notes |
|--------|-------------|-------|
| default (combo_kernels=True, cdt=True) | 33.6 | Best |
| multi_kernel=2 | 61.14 | Much worse |
| multi_kernel=3 | 58.76 | Much worse |

## Kernel count
- Inductor: 2 kernels (persistent reduction + pointwise epilogue)
- Oracle: 3 kernels (partial reduce + finalize + slice-add epilogue)

## Status: residual gap (1.098x), known design limitation

The gap is under 1.1x and results from the same SCHEDULER_FUSION limitation documented in sum_sum_02744d87feff and other DenseNet BN-backward repros. The scheduler cannot form a single multi-output reduction plan that shares the masked/centered producer, sinks the dependent side output to the live channel slice, and keeps the residual slice-add chain in one coordinated plan.

This is the same family as sum_sum_02744d87feff (which was fixed from 2.28x to 1.13x via the split-K threshold tightening). The remaining ~10% gap is the inherent cost of not having a fully cooperative multi-output reduction template.

## Fix direction (design doc)

To close the remaining gap, Inductor would need:
1. A scheduler enhancement to detect sibling reductions with shared masked producers
2. Codegen support for dependent epilogues that consume finalized reduction results
3. The ability to sink the slice-limited output computation into the epilogue rather than materializing the full per-channel intermediate

This affects multiple DenseNet BN-backward repros in the corpus.

## File references
- Oracle: repros/canonical/sum_sum_ed9e2feedf2a/oracle_densenet_bn_tail.py
- Model: torchbench_densenet121_train_001
- Pattern: Dual sibling reductions + 20 residual slice-adds + BN grad epilogue (C=384)
- Related: sum_sum_02744d87feff (same family, fixed from 2.28x to 1.13x)
- Scheduler: /tmp/pytorch-work/torch/_inductor/scheduler.py (can_fuse, score_fusion)
