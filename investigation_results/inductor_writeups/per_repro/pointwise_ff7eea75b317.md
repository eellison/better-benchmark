# pointwise_ff7eea75b317

## Classification: MULTI_OUTPUT_SIBLING_LAYOUT_FUSION

## Current Result

- Family: `visformer_qkv_layout`
- Oracle path: `repros/canonical/pointwise_ff7eea75b317/oracle_visformer_qkv_layout.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle directly materializes the complete Visformer Q/K/V layout split from the original contiguous NCHW f32 tensor into the three fresh contiguous outputs in one Triton multi-output layout kernel. Inductor lowers the sibling unbind/permute/clone/view materializations as separate generic layout-copy work because its scheduler does not fuse multiple users of an unbound strided view when the users need different final output index maps.

## Root cause

The repro takes one `[B, 3*H*D, Hp, Wp]` convolution output, views it as `[B, 3, H, D, P]`, and then unbinds along dim=1 to produce Q, K, V with different permute/clone/view layouts:
- Q: `[B*H, P, D]` (contiguous)
- K: `[B*H, D, P]` (transposed layout)
- V: `[B*H, P, D]` (contiguous)

Inductor schedules these as 3 separate layout copy kernels, each reading the full input once. The oracle reads the input once and writes all three outputs from a single 3D grid kernel (row x d_block x p_block), saving 2/3 of the input read traffic.

## Kernel count

- Oracle: 1 kernel (_qkv_contiguous_layout_kernel, 3D grid, multi-output)
- Inductor: 3 kernels (one layout copy per Q/K/V output)

## Config exploration

| Config | Expected Impact |
|--------|----------------|
| combo_kernels=True | May help if it recognizes the sibling pattern |
| multi_kernel | Not applicable (no reductions) |
| coordinate_descent_tuning | Only tunes existing kernel tile sizes |

combo_kernels might partially address this by scheduling the three layout copies as sub-kernels in a combo group, but it would still read the input 3 times unless the scheduler recognizes the shared-source multi-output pattern.

## Recommendation

Teach layout-copy scheduling to emit a fused multi-output producer for sibling Q/K/V materializations with per-output affine store maps. When multiple consumers of a shared strided view each need separate contiguous clones, the scheduler should emit one kernel that iterates over the shared source and writes to multiple output buffers with different index maps.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (can_fuse, score_fusion for sibling layout copies)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (multi-output layout kernel emission)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (unbind/view/clone detection)
