# sum_sum_sum_3814c418da03
## Compile: 36us, Oracle: 35us, Gap: 1.04x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor's combo_kernels + CD tuning nearly matches the cooperative split-K oracle for this Swin MLP/layer-norm-backward shape (M=6272, C=1024); the 4% gap is within measurement noise, indicating Inductor's fused persistent kernel already captures most of the fusion benefit on this hardware.
## Fix path: No significant Inductor change needed; gap is within noise. Cooperative split-K template could provide marginal improvement but is not required for this shape.
## Status: at_floor
