# var_mean_var_mean_0795d88a1e39

## Compile: 188.19us, Oracle: 158.62us, Gap: 1.186x

## Diagnosis: DUAL_LAYERNORM_PARTITION_FUSION

## Root cause: Same as var_mean_var_mean_7eb8a9047c3e. The oracle computes a complete Swin dual channel-LayerNorm region (two sequential var_mean reductions, affine epilogues, and 7x7 window partition clone/view) in a single kernel by keeping intermediate normalized values in registers. Inductor lowers this as 3 separate kernels because it cannot fuse two dependent reductions with an interleaved layout permutation.

## Kernel count
- Inductor: 3 kernels (permute_var_mean, add_mul_permute_rsqrt_sub_var_mean, poi_fused_add_clone_mul_permute_rsqrt_sub_var_mean_view)
- Oracle: 1 kernel (dual layernorm + window partition in one pass)

## Config exploration results
- multi_kernel=1 (default): 188.19us (ratio 1.186x)
- multi_kernel=2: 187.17us (ratio 1.181x) - no improvement
- multi_kernel=3: 186.34us (ratio 1.175x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: DUAL_LAYERNORM_PARTITION_FUSION

Identical pattern to var_mean_var_mean_7eb8a9047c3e. Two chained LayerNorm reductions with window-partition layout change. Inductor scheduler cannot fuse dependent reduction chains into one pass.

## Fix path
Same as var_mean_var_mean_7eb8a9047c3e: scheduler enhancement to detect and fuse chained LayerNorm reductions with register-resident forwarding. File: `/tmp/pytorch-work/torch/_inductor/scheduler.py`

## Status: design_doc
