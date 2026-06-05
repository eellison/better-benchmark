# sum_sum_sum_eb29178ac0d2

## Classification: SCHEDULER_FUSION

## Benchmark Results
- Oracle: 155.42 us
- Inductor compiled: 284.61 us
- Ratio: 1.831x

## Kernel Count
- Inductor: 1 kernel (single fused kernel but with suboptimal schedule)
- Oracle: 3 kernels (dual-branch partial reduction + finalizer + epilogue)

## Root Cause

The oracle computes the complete ShuffleNet channel-shuffle/cat/permute scope with both sibling batch-norm-backward-style branches. The repro structure:

1. A `view` as [N, C=116, H, W] (the concatenated input)
2. A channel shuffle: view as [N, C=58, 2, H, W] -> permute -> contiguous -> view back
3. Two BN-backward branches on the even/odd 58-channel halves:
   - Each branch: pointwise BN-backward -> channel reduction sum([0,2,3]) -> broadcast epilogue
4. Four outputs: two [512, 58, 28, 28] tensors + two [58] vectors

The oracle strategy:
1. `_dual_branch_partial_kernel`: reads the shuffled view directly (computing the shuffle as index arithmetic), shares the two branch reductions in one pass, accumulates partial sums for both branches
2. Finalizer: sums the partials to get the [58] vectors
3. Epilogue: writes the [512, 58, 28, 28] outputs using the finalized summaries

Inductor produces 1 kernel but it treats the channel shuffle (cat/view/permute/clone) as a layout operation that must be scheduled, followed by the two BN-backward branches as dependent regions. Even though combo_kernels fuses them into one kernel, the internal schedule within that kernel is suboptimal - it doesn't exploit the virtual nature of the channel shuffle.

Shape info: N=512, C=58, H=28, W=28. Total reduction per channel = 512*28*28 = 401,408 elements.

## Config Exploration
- `combo_kernels=True, coordinate_descent_tuning=True`: 284.61 us (baseline)
- The gap is structural - the oracle's key advantage is treating the channel shuffle as virtual index arithmetic rather than materializing it

## What Inductor Needs (Design Doc)

**Enhancement needed**: Scheduler fusion that keeps ShuffleNet channel-shuffle cats virtual and sinks the paired BN-backward reductions into one full-scope plan.

The scheduler would need to:
1. Recognize cat/view/permute/clone sequences as "virtual producers" that don't need materialization
2. Represent the channel shuffle as strided index arithmetic in the downstream reduction
3. Fuse the two BN-backward branches (sharing the interleaved channel indices) into one multi-output reduction template
4. Emit tensor + vector epilogues from the finalized summaries

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - virtual producer recognition for cat/permute patterns
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - dual-branch reduction codegen with shared virtual indexing
- `/tmp/pytorch-work/torch/_inductor/ir.py` - possible enhancement to represent virtual shuffles

**Affected repro count**: ShuffleNet variants and any architecture with channel-shuffle + per-branch BN-backward. Likely 3-5 repros in the corpus.
