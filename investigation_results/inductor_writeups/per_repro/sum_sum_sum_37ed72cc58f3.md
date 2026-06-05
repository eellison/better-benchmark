# sum_sum_sum_37ed72cc58f3

## Compile: 244.77us, Oracle: 120.77us, Gap: 2.027x

## Classification: COOPERATIVE_SPLIT_K

## Root Cause

The oracle computes the complete Swin Transformer window-reverse / indexed layernorm-backward / drop-path return tuple in a coordinated cooperative split-K multi-output kernel. It applies dynamic height/width index gather (torch.roll emulation via index), reduces each 256-wide row for the input-gradient epilogue, writes the [256, 100352] transposed side output, and accumulates all three [256] column reductions from the same row-tiled producer.

Inductor generates 6 kernels:
1. `triton_per_fused_add_clone_index_mul_permute_sub_sum_view_0`: cooperative reduction with workspace
2. `triton_mor_finalize_sum_1`: finalize cooperative sum
3. `triton_red_fused_clone_index_permute_sum_view_2`: partial column reduction for transpose output
4. `triton_poi_fused_convert_element_type_div_mul_view_3`: drop-path pointwise
5. `triton_red_fused_convert_element_type_div_mul_sum_view_4`: partial reduction
6. `triton_red_fused_convert_element_type_div_mul_sum_view_5`: final reduction

The fundamental issue is that Inductor cannot coordinate all these operations into a single producer pass: the window layout reconstruction (index gather), row-local reductions for layernorm backward, dependent transpose side-output store, drop-path scaling, and sibling column accumulators all need to share one coordinated pass over the data.

## Kernel Count
- Oracle: 1-2 kernels (coordinated cooperative split-K)
- Inductor: 6 kernels (separate reductions, layout ops, and pointwise)

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo_kernels + CDT | 244.77 | baseline (2.027x) |
| combo + multi_kernel=2 | 258.27 | worse |
| combo + multi_kernel=3 | 254.18 | worse |

No config changes help. This is a complex structural limitation.

## Fix Assessment: Design doc

This is a complex multi-output reduction + indexed layout reconstruction pattern from Swin Transformer training. The oracle achieves its speedup by:
1. Applying the window-reverse index gather inline (no materialized intermediate)
2. Computing layernorm backward row reductions from the gathered data
3. Writing the transposed side output in the same pass
4. Accumulating three [256] column sums atomically

### What's needed:
The scheduler needs a cooperative split-K template that coordinates:
- Dynamic indexed layout reconstruction (window-reverse via index)
- Row-local reductions (layernorm backward style: sum(x*weight), sum(x))
- Dependent pointwise epilogue (sub, mul, div for drop-path)
- Transposed side-output store
- Multiple sibling column reductions with atomic accumulation

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: cannot fuse index -> reduce -> permute -> reduce chain
- `/tmp/pytorch-work/torch/_inductor/ir.py`: realize_hint on indexed intermediates
- `/tmp/pytorch-work/torch/_inductor/choices.py`: cooperative reduction strategy

### Affected repro count:
This pattern is specific to Swin Transformer training with window attention. Likely 3-5 repros in the corpus with the same cooperative split-K classification.

## Details
- Model: timm_swin_base_patch4_window7_224 (train)
- Shape: [100352, 256] f32 (= 128 * 784 tokens * 256 channels)
- Pattern: window-reverse(index) -> layernorm_bwd(sum, mul, sub) -> drop_path(mul, div) -> permute [256, 100352] + 3x sum(dim=0) [256]
- This is a 2x gap, one of the largest in the corpus
