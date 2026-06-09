# var_mean_mean_626fb0807512

## Summary
- **Model**: torchbench_phlippe_densenet_train_000 (DenseNet BN + ReLU + spatial mean)
- **Pattern**: cat([f32[128,16,4,4], f32[128,168,4,4]], dim=1) -> per-channel var_mean over [128, 184, 4, 4], running-stat copy_, affine, ReLU, spatial mean -> [128, 184]
- **Oracle**: 6.624 us (1 kernel: reads directly from two cat inputs without materializing concatenation)
- **Compile (CDT)**: 8.672 us (2 kernels)
- **Ratio**: 1.309x
- **Classification**: CAT_MATERIALIZATION

## Root Cause

The oracle avoids materializing the concatenated f32[128, 184, 4, 4] tensor entirely. Instead, it reads directly from the two cat inputs (f32[128,16,4,4] and f32[128,168,4,4]) by branching on `channel < c0` within the kernel. It computes BN statistics, running-stat updates, normalization, ReLU, and spatial mean all in one kernel without ever writing the concatenated buffer.

Inductor generates 2 kernels:
1. A reduction kernel that computes per-channel var_mean statistics -- this must either materialize the cat or handle the split inputs
2. A pointwise+reduction kernel for normalization + ReLU + spatial mean

The 31% gap comes from:
1. **Cat materialization** - Inductor materializes the full f32[128, 184, 4, 4] concatenation as an intermediate buffer (128*184*16*4 = ~5.7 MB write + read)
2. **2 kernel launches** vs 1 (significant overhead at this tiny 6.6us scale)
3. **Small tensor** - at 6.6us total, kernel launch overhead accounts for a large fraction

With `multi_kernel=3` the harness reports 8.8us (still 1.33x). The gap is fundamentally from cat materialization + extra kernel launch, not tile selection.

## Kernel Count
- Oracle: 1 kernel (reads two cat inputs directly, fuses everything)
- Inductor: 2 kernels (cat + reduction, then normalize/ReLU/pool)

## Config Exploration
| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| CDT (default) | 8.672 | 1.309x |
| CDT + combo + multi_kernel=2 | ~12.3* | ~1.86x |
| CDT + combo + multi_kernel=3 | 8.800 | 1.328x |

*multi_kernel=2/3 do not help and sometimes hurt at this scale due to the overhead of trying multiple reduction strategies for tiny tensors.

## Design Doc

The fundamental issue is that Inductor materializes the `cat` before computing the reduction. The oracle reads directly from the two input tensors, branching by channel index (if channel < 16: read from x0, else: read from x1 at offset channel-16).

To fix this, Inductor needs a "cat-through-reduction" pass (similar to the existing cat-through-reduction pass for simpler cases) that:
1. Recognizes `cat(inputs, dim=C) -> var_mean(over [0,2,3])` pattern
2. Eliminates the cat materialization by teaching the reduction kernel to read from multiple source buffers
3. Fuses the normalization epilogue with the stats computation into a single kernel

For the BN-training pattern specifically, the norm template needs to handle cat inputs natively.

### Relevant files
- `/tmp/pytorch_work/torch/_inductor/fx_passes/post_grad.py` - cat elimination passes
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion decisions with cat producers
- `/tmp/pytorch-work/torch/_inductor/kernel/norm.py` - norm template for training BN

### Affected repros
The DenseNet pattern (cat -> BN -> activation -> pool) appears in all DenseNet blocks. Similar repros likely exist with different channel counts. This is a known pattern: see AGENT_INSTRUCTIONS.md which mentions "CAT_MATERIALIZATION: cat() materializes large intermediate that could be avoided."

## Status: design_doc (requires cat-through-reduction fusion + norm-template enhancement)
