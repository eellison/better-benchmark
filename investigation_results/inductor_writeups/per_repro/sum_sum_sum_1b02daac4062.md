# sum_sum_sum_1b02daac4062
## Compile: 36us, Oracle: 44us, Gap: 0.81x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor's combo_kernels + CD tuning already achieves better latency than the cooperative split-K oracle on this Swin layer-norm-backward/drop-path shape (M=6272, C=1024); Inductor's persistent kernel avoids the cooperative tile coordination overhead that the oracle pays for finalization of partial column accumulators.
## Fix path: No Inductor change needed; compile already beats the oracle floor by 19%. Oracle needs re-tuning for this hardware or the cooperative approach is suboptimal at this M/C ratio.
## Status: at_floor
