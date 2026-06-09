# sum_sum_sum_da376d09d2cb
## Compile: 37us, Oracle: 44us, Gap: 0.84x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor's combo_kernels + CD tuning already achieves better latency than the cooperative split-K oracle on this Swin stochastic-depth layernorm-backward shape; the oracle's row-tiled approach with column accumulation finalization adds overhead that Inductor's fused persistent kernel avoids at this tile size (M=6272, C=1024).
## Fix path: No Inductor change needed; compile already beats the oracle floor. Oracle may need re-tuning for this hardware.
## Status: at_floor
