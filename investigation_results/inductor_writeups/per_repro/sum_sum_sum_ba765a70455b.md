# sum_sum_sum_ba765a70455b

## Summary
- **Model**: timm_swin_base_patch4_window7_224_train_001
- **Pattern**: Swin window-reverse + LayerNorm backward + drop-path + column reductions
- **Ratio**: 1.426x (oracle 50.14us vs compile 71.52us)
- **Classification**: COOPERATIVE_SPLIT_K

## Root Cause

The repro computes the Swin Transformer's window-reverse layer-norm-backward with drop-path:
- Input: `mm_146 [25088, 512]` (windowed attention output)
- Window reverse: view/permute/clone to [128, 14, 14, 512]
- Row-local LayerNorm backward: sum(x*weight, dim=-1), sum(x*weight*rhs, dim=-1) per row
- Scale by `arg529_1` and apply drop-path mask
- Column reductions: sum([0,1,2]) for two [512] gradient vectors
- Side output: permuted/transposed [512, 25088] tensor

The oracle uses a single cooperative row-tile kernel that:
1. Reconstructs window layout on-the-fly (avoiding the permute/clone materialization)
2. Computes row-local LayerNorm backward scalars
3. Writes the full gradient tensor and transposed side output
4. Atomically accumulates three [512] column reduction partials
5. A small finalization kernel sums the partials

Inductor generates 5 kernels:
1. `triton_per_fused_clone_mul_permute_sum_view_0`: cooperative reduction for row sums over [25088, 512] (xnumel=25088, r0_numel=512)
2. `triton_mor_finalize_sum_1`: finalize first sum
3. `triton_mor_finalize_sum_2`: finalize second sum
4. `triton_poi_fused_add_clone_convert_element_type_div_mul_permute_sub_view_3`: pointwise epilogue (xnumel=12845056)
5. `triton_red_fused_add_clone_convert_element_type_div_mul_permute_sub_sum_view_4`: column reduction (xnumel=100352, r0_numel=128)

The gap comes from:
1. **Redundant data reads**: The input [25088, 512] is read in kernel 0 (row reductions), again in kernel 3 (pointwise epilogue), and again in kernel 4 (column reduction). The oracle reads it once.
2. **Window layout materialization**: Inductor materializes the permute/clone [128, 2, 7, 2, 7, 512] -> [128, 14, 14, 512] as an explicit memory copy in the fused kernel. The oracle reconstructs window coordinates on-the-fly with index arithmetic.
3. **Separate column reduction**: The column sum requires re-reading the gradient tensor (12.8M elements) that was just computed in kernel 3.

## Kernel Count
- **Inductor**: 5 kernels (cooperative row reduction + 2 finalizers + pointwise + column reduction)
- **Oracle**: 2 kernels (cooperative row-tile with atomic column accumulation + finalization)

## Config Exploration
- `combo_kernels=True`: Enabled, but the 5 kernels have different grid shapes and cannot be combined
- `coordinate_descent_tuning=True`: Enabled, helps tune block sizes within each kernel
- `triton.multi_kernel=2`: 77.93us (slightly worse than default 71.52us)
- `triton.multi_kernel=3`: 79.20us (slightly worse than default)
- Best config is default (multi_kernel=0): 71.52us. multi_kernel modes don't help here.
- The row-local reductions are already cooperative (25088 rows over 512 elements), which is good. The gap is that the column reductions and pointwise epilogue are separate passes.

## Design Doc

The 42% gap is a **multi-pass data read** issue. The oracle computes everything (row stats, column partials, gradient output, transposed output) in a single pass over the input data. The fix requires:

1. **Fuse pointwise epilogue with dependent column reduction**: When a pointwise kernel's output is immediately reduced along a different axis, fuse the reduction accumulation into the same kernel (each tile atomically adds its partial).
2. **Window layout fusion**: The view/permute/clone pattern for Swin windows should be reconstructed via index math rather than materialized.
3. **Multi-output row-tile template**: Allow a single kernel to produce the full gradient tensor, the transposed side output, and accumulate column partials in one pass.

**Relevant files:**
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: fusion of pointwise with dependent reduction
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: atomic accumulation in non-reduction kernels
- `/tmp/pytorch-work/torch/_inductor/ir.py`: window-reverse pattern recognition

## Affected Repro Count
The COOPERATIVE_SPLIT_K with dependent column reduction pattern appears in 20+ Swin/ViT training repros. The window-reverse index fusion alone affects 8+ Swin-specific repros.

## Re-measurement (2026-06-08)

- Oracle: 48.74 us
- Compiled: 71.84 us
- Ratio: 1.474x (essentially unchanged from 1.426x, within measurement noise)

The split-K improvements did not help this case. The gap here is fundamentally about multi-pass
data reads (fusing pointwise epilogue with dependent column reduction), not split-K factor tuning.
