# sum_sum_a114cf26ba31

## Compile: 29.5us, Oracle: 28.74us, Gap: 1.027x

## Diagnosis: SCHEDULER_FUSION (residual gap)

## Root cause

The oracle computes the full DenseNet BN-backward tail by sharing the ReLU-mask `where`, channel centering, and two sibling per-channel reductions across all C=544 channels, then uses the finalized reductions in the dependent epilogue for live channels 512:544 while adding all 15 residual slices. Inductor schedules a fused reduction + a pointwise epilogue as 2 kernels, which is nearly optimal.

The gap of 1.027x is within measurement noise and at floor. The pattern is the same DenseNet BN-backward family as other sum_sum repros but with C=544 and 15 residual slices. Inductor's 2-kernel strategy (persistent reduction + pointwise) achieves near-oracle performance.

## Config exploration

| Config | Compile (us) | Notes |
|--------|-------------|-------|
| default (combo_kernels=True, cdt=True) | 29.5 | Best |

## Kernel count
- Inductor: 2 kernels (persistent reduction + pointwise epilogue)
- Oracle: 3 kernels (partial reduce + finalize + epilogue)

## Status: AT_FLOOR

The 1.027x gap is within noise. No fix justified.

## File references
- Oracle: repros/canonical/sum_sum_a114cf26ba31/oracle_multi_output_reduction.py
- Model: torchbench_densenet121_train_001
- Pattern: Dual sibling reductions + sliced residual add epilogue (C=544, 15 residuals)
