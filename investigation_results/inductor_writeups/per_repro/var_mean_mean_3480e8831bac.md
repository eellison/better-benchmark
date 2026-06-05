# var_mean_mean_3480e8831bac

## Compile: 37.82us, Oracle: 22.27us, Gap: 1.698x

## Diagnosis: NORM_SPATIAL_MEAN_FUSION_WITH_STOCHASTIC_PRODUCER

## Root cause: The oracle computes the complete Swin drop-path residual add, hidden-size-1024 LayerNorm, rsqrt/1024 side output, and final 7x7 spatial mean directly from the source tensors in 2 kernels (a spatial-chunked norm kernel + a finalize spatial-mean kernel). Inductor schedules this as 3 kernels: (1) stochastic drop-path seed lookup, (2) var_mean+affine per-token normalization, (3) spatial mean reduction consuming the materialized normalized output. The key gap is that Inductor materializes the full [128,49,1024] normalized intermediate before the spatial-mean reduction, while the oracle accumulates the spatial mean on-the-fly during normalization.

## Kernel count
- Inductor: 3 kernels (poi_fused_inductor_lookup_seed_inductor_random, per_fused_add_convert_element_type_div_lt_mul_rsqrt_var_mean_view, per_fused_add_convert_element_type_div_lt_mean_mul_rsqrt_sub_var_mean_view)
- Oracle: 2 kernels (spatial-chunk norm kernel, finalize spatial-mean kernel)

## Config exploration results
- multi_kernel=1 (default): 37.82us (ratio 1.698x)
- multi_kernel=2: 37.95us (ratio 1.699x) - no improvement
- multi_kernel=3: 37.95us (ratio 1.699x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: NORM_SPATIAL_MEAN_FUSION_WITH_STOCHASTIC_PRODUCER

The oracle eliminates the materialized normalized activation by accumulating the spatial mean as a side-effect of the normalization pass. Additionally, the stochastic drop-path (Inductor RNG) producer is inlined into the reduction kernel rather than being a separate seed-lookup kernel.

## Fix path
Two enhancements needed in `/tmp/pytorch-work/torch/_inductor/scheduler.py`:
1. Fuse Inductor RNG seed-lookup producers into downstream reduction consumers (eliminate kernel 1)
2. Allow normalization reduction templates to accumulate a downstream orthogonal-axis mean on-the-fly, avoiding materialization of the normalized intermediate

This is a superset of the general "reduction epilogue feeds another reduction" pattern where materializing the intermediate is wasteful.

## Status: design_doc
