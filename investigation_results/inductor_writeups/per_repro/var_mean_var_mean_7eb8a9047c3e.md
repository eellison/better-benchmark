# var_mean_var_mean_7eb8a9047c3e

## Compile: 188.19us, Oracle: 158.53us, Gap: 1.187x

## Diagnosis: DUAL_LAYERNORM_PARTITION_FUSION

## Root cause: The oracle computes a complete Swin dual channel-LayerNorm region (two sequential var_mean reductions, affine epilogues, and 7x7 window partition clone/view) by scattering source spatial rows directly to the final [401408,128] window layout in a single kernel, whereas Inductor lowers this as 3 separate kernels: (1) first var_mean reduction with permute, (2) second var_mean + affine + permute, (3) final pointwise clone/sub/add/mul with permute. The oracle keeps intermediate normalized values in registers, avoiding materialization between the two layernorm passes and the final layout scatter.

## Kernel count
- Inductor: 3 kernels (triton_red_fused_permute_var_mean_0, triton_red_fused_add_mul_permute_rsqrt_sub_var_mean_1, triton_poi_fused_add_clone_mul_permute_rsqrt_sub_var_mean_view_2)
- Oracle: 1 kernel (dual layernorm + window partition in one pass)

## Config exploration results
- multi_kernel=1 (default): 188.19us (ratio 1.187x)
- multi_kernel=2: 187.68us (ratio 1.184x) - no improvement
- multi_kernel=3: 188.22us (ratio 1.187x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: DUAL_LAYERNORM_PARTITION_FUSION

The oracle demonstrates that chaining two layernorm reductions with a window-partition layout change can be done in one kernel pass. Inductor's scheduler cannot fuse two dependent reduction nodes (where the second reduction reads the output of the first) with an interleaved layout permutation into a single kernel.

## Fix path
This requires a specialized scheduler enhancement in `/tmp/pytorch-work/torch/_inductor/scheduler.py` to detect chained LayerNorm reductions where the intermediate is only consumed by the next normalization, enabling register-resident forwarding. The window-partition scatter (view/permute/clone) should be sunk into the final store of the fused kernel.

Related patterns: var_mean_var_mean_0795d88a1e39 (identical shapes/pattern)

## Status: design_doc
