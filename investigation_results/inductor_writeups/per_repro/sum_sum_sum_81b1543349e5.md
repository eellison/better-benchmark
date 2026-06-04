# sum_sum_sum_81b1543349e5
## Compile: 35us, Oracle: 44us, Gap: 0.80x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor's combo_kernels + CD tuning already achieves better latency than the cooperative split-K oracle on this Swin layer-norm-backward/drop-path shape (M=6272, C=1024); the oracle's cooperative tile finalization and transposed side-output store add overhead that Inductor's fused persistent approach avoids.
## Fix path: No Inductor change needed; compile already beats the oracle floor by 20%. Oracle needs re-tuning with more aggressive tile parameters for this hardware.
## Status: at_floor
