# var_mean_mean_2ac1c2eb8544

## Compile: 36.26us, Oracle: 24.7us, Gap: 1.468x

## Diagnosis: NORM_SPATIAL_MEAN_FUSION

## Root cause: The oracle computes the complete Swin residual-add, hidden-dimension variance/mean, layernorm affine, and 7x7 spatial mean by staging only per-token layernorm statistics and reducing directly to the final [128,1024] output. Inductor materializes the full normalized [128,49,1024] intermediate before launching a separate spatial mean reduction. The oracle uses a two-kernel approach: (1) compute per-row mean/invstd stats, (2) a fused kernel that reads the raw input, applies normalization using pre-computed stats, applies affine, and accumulates the spatial mean across the 49 spatial positions -- all without materializing the 128*49*1024 normalized activation.

## Kernel count
- Inductor: 2 kernels (var_mean + affine LayerNorm, then spatial mean reduction)
- Oracle: 2 kernels (row stats kernel, then spatial-mean-with-affine kernel that avoids materializing normalized activation)

## Config exploration results
- multi_kernel=1 (default): 36.26us (ratio 1.468x)
- multi_kernel=2: 36.19us (ratio 1.473x) - no improvement
- multi_kernel=3: 35.84us (ratio 1.451x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: NORM_SPATIAL_MEAN_FUSION

The key insight is that the oracle avoids materializing the normalized intermediate. Instead of: read -> normalize -> write -> read -> spatial_mean -> write, the oracle does: compute stats once, then read raw data -> normalize-in-registers -> accumulate spatial mean -> write final [128,1024]. This eliminates a full 128*49*1024*4 = 25.6MB round-trip.

## Fix path
Scheduler enhancement in `/tmp/pytorch-work/torch/_inductor/scheduler.py` to detect when a normalization output's only consumer is a downstream spatial-axis mean. In this case, the scheduler should generate a fused kernel that:
1. Reads pre-computed stats (mean, invstd)
2. Reads raw input
3. Normalizes in registers
4. Accumulates spatial reduction
5. Writes only the final reduced [batch, hidden] output

This avoids materializing the full [batch, spatial, hidden] normalized activation.

Related: var_mean_mean_3480e8831bac (same pattern with added stochastic producer)

## Status: design_doc
