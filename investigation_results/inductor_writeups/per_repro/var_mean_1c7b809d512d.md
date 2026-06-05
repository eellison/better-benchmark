# var_mean_1c7b809d512d

## Compile: 10.05us, Oracle: 6.05us, Gap: 1.661x

## Diagnosis: ALGEBRAIC_ELIMINATION

## Root cause: The oracle exploits dead computation elimination. The repro computes residual add + LayerNorm over ALL 32*197=6304 rows of a ViT [32,197,384] tensor, then selects only token 0 (CLS token) from each batch -- yielding just 32 output rows of shape [32,384]. The oracle recognizes that only 32 rows contribute to the output and reduces only those rows, avoiding 99.5% of the computation (32 vs 6304 rows).

Inductor emits 2 kernels:
1. `triton_per_fused_add_convert_element_type_var_mean_view_0` -- reduces all 6304 rows to compute mean/variance
2. `triton_poi_fused_add_clone_convert_element_type_mul_rsqrt_select_sub_var_mean_view_1` -- applies affine + select + clone

The oracle processes only 32 rows (one per batch item, at token index 0), yielding a ~197x reduction in compute and memory traffic for the reduction kernel.

## Kernel count
- Inductor: 2 kernels (var_mean reduction over 6304 rows, then pointwise affine+select)
- Oracle: 1 kernel (reduces only 32 CLS-token rows)

## Config exploration results
- multi_kernel=0: 63.93us, multi_kernel=1: 51.63us, multi_kernel=2: 50.94us, multi_kernel=3: 51.03us
- multi_kernel helps slightly but cannot fix the fundamental 197x row over-computation
- coord_descent_tuning already enabled

## Fix path: ALGEBRAIC_ELIMINATION in the scheduler or FX passes. The optimizer needs to:
1. Detect that `select(dim=1, index=0)` follows a row-independent operation (LayerNorm reduces over dim=2, preserving row independence)
2. Push the select backward through the LayerNorm graph: since LayerNorm over dim=2 is row-local, selecting a subset of rows before vs after is equivalent
3. Generate the narrowed computation: only compute add + var_mean + affine for the 32 selected rows

This is a significant optimization (1.66x) and generalizes to any CLS-token extraction pattern in ViTs. The key insight is that row-local reductions commute with row-selection.

## Status: design_doc

## Details
- Model: torchbench_timm_vision_transformer_infer_000
- Pattern: addmm [6304,384] f16 -> view [32,197,384] -> add(residual) -> f32 cast -> var_mean([2], correction=0) -> LayerNorm(eps=1e-6) -> f16 cast -> select(dim=1, index=0) -> clone -> output [32,384] f16
- Shapes: Full computation [32,197,384], but only [32,384] output matters
- Compute savings: 197x fewer rows to reduce (32 vs 6304)
- Memory savings: avoids reading/writing [32,197,384] intermediate (~15.5MB in f32)
- File references: /tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py (select-through-layernorm), /tmp/pytorch-work/torch/_inductor/scheduler.py (dead row elimination)
- This pattern is common across ALL ViT inference workloads that extract CLS tokens
