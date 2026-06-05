# var_mean_mean_188f42114f68

## Compile: 11.2us, Oracle: 9.86us, Gap: 1.136x

## Diagnosis: BN_TRAINING_MULTI_CONSUMER_FUSION

## Root cause: The oracle computes the full ResNet training-BatchNorm + residual ReLU scope in one channel kernel, returning the BN rsqrt side output, spatial ReLU mean view, le mask, squeezed mean view, and both in-place running-stat copy_ outputs while avoiding a materialized normalized activation. Inductor schedules this as 2 kernels: (1) BN-training var_mean + running-stat copy_ updates + rsqrt, (2) residual add + ReLU + le mask + spatial mean. The gap comes from the second kernel re-reading the full activation that the first kernel already processed.

## Kernel count
- Inductor: 2 kernels (red_fused_add_copy__mul_rsqrt_squeeze_var_mean, per_fused_add_le_mean_mul_relu_sub_unsqueeze)
- Oracle: 1 kernel (channel-tiled BN-train + residual ReLU + le + spatial-mean)

## Config exploration results
- multi_kernel=1 (default): 11.2us (ratio 1.136x)
- multi_kernel=2: 11.26us (ratio 1.143x) - no improvement
- multi_kernel=3: 11.46us (ratio 1.159x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: BN_TRAINING_MULTI_CONSUMER_FUSION

Inductor's BN-training template produces mean/invstd/running-stat updates but does not sink the downstream residual-add, ReLU, le-mask, and spatial-mean consumers into the same channel-tiled reduction. The scheduler cannot fuse a normalization reduction producer with multiple downstream pointwise and reduction consumers while preserving mutable copy_ side outputs.

## Fix path
Enhancement needed in `/tmp/pytorch-work/torch/_inductor/scheduler.py`: Allow the BN-training template to expose mean/rsqrt/running-stat outputs AND sink affine+residual+ReLU+le+spatial-mean consumers into the same channel-tiled reduction schedule. This eliminates the re-read of the normalized activation.

## Status: design_doc
