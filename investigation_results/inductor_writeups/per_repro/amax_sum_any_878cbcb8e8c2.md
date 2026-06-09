# amax_sum_any_878cbcb8e8c2

## Compile: 252.77us, Oracle: 251.55us, Gap: 1.005x

## Diagnosis: AT_FLOOR

## Root cause: The oracle fuses the complete T5 attention softmax/dropout (view [96,1024,1024] -> [8,12,1024,1024], fp32 bias add, stable last-dimension softmax, Inductor RNG dropout, scale, expand/view, and final [96,1024,1024] permuted output layout) into a single persistent row-softmax Triton kernel. Inductor already achieves essentially the same performance with its fused kernel -- ratio is 1.005x.

## Status: closed_at_floor

## Details

- Model: torchbench_hf_T5_base_train (full attention softmax + dropout)
- Pattern: view -> add -> amax -> sub -> exp -> sum -> div -> inductor_random dropout -> scale -> expand -> view -> permute
- Shape: [96, 1024, 1024] with K=1024 (row length), 96K rows
- Inductor kernels: 1 (fused persistent reduction with softmax + dropout)
- Oracle kernels: 1 (softmax + dropout + output transpose)
- The 0.5% gap is within measurement noise. No actionable fix needed.
- Stochastic: yes (inductor_random dropout, correctness check skips output comparison)
- Config exploration: not needed -- already at floor

## Classification: AT_FLOOR

Inductor matches oracle performance. No investigation needed.
