# pointwise_6d842b54b40d — oracle_cat_bn_relu

## Summary
Oracle: 144.19us | Compiled: 256.8us (default), 175us (best config) | Ratio: 1.781x (default), 1.21x (best) | Status: GOOD

## Classification: SCHEDULER_FUSION (cat-as-output-layout)

## Root cause
The oracle computes 4-branch batch normalization + ReLU and channel concatenation in a single kernel using a 2D grid where `program_id(1)` indexes the branch (0-3). Each branch writes directly into its destination channel slice of the output [128, 768, 17, 17] tensor, eliminating the need for a separate concat/copy pass.

Inductor currently fuses the cat operation into a single pointwise kernel over the entire 28M-element output, but uses 1D flattened indexing with branch conditionals (`tmp0 < 192`, `tmp0 >= 192 && tmp0 < 384`, etc.). This means:
1. Every thread evaluates all 4 branch conditions even though it only executes one
2. Each branch has its own data loading logic guarded by masks, leading to divergent control flow
3. The per-channel BN parameters are loaded with the same conditional masking, reducing efficiency
4. The 1D flattened indexing requires expensive div/mod operations to extract channel and spatial indices

The oracle's 2D grid approach avoids all conditionals - each warp knows exactly which branch it's processing and uses simple affine indexing.

## Kernel count
- Inductor: 1 kernel (fused pointwise with branch conditionals for cat)
- Oracle: 1 kernel (2D grid, branch as grid dim)

## Config exploration
| Config | Time (us) | Ratio |
|--------|-----------|-------|
| coordinate_descent_tuning=True, combo_kernels=True (default bench) | 256.8 | 1.781x |
| coordinate_descent_tuning=True only | 175.0 | 1.214x |
| coordinate_descent_tuning=True, combo_kernels=True, multi_kernel=3 | 175.1 | 1.214x |
| coordinate_descent_tuning=True, combo_kernels=True, multi_kernel=1 | 382.5 | 2.653x |
| max_autotune_pointwise=True | 309.9 | 2.149x |

Note: combo_kernels appears to HURT here (256.8 vs 175us without it), likely because it changes the XBLOCK selection for this branchy kernel.

## File/line references
- `/tmp/pytorch-work/torch/_inductor/ir.py:6496` — ConcatKernel: currently implemented as NopKernel that just adjusts storage layout offsets
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` — fusion decisions; cat consumers not modeled as destination-slice epilogues
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` — 1D pointwise codegen with branch conditionals

## Design doc: Why it can't be fixed today

The fundamental issue is that Inductor's `ConcatKernel` is a `NopKernel` - it doesn't generate any code itself, it just arranges for upstream producers to write into slices of a shared output buffer. When those producers are pointwise kernels, they each get their own kernel launch or get fused together into a single kernel with conditionals.

**What's needed**: A scheduler enhancement to recognize patterns where:
1. Multiple sibling pointwise producers all feed into a single cat operation
2. The cat is along a dimension where each producer has a fixed, known extent
3. The producers can be emitted as a single kernel with a multi-dimensional grid where one grid axis indexes the "branch" (cat input index)

This would eliminate branch conditionals and allow each sub-program to use simple affine indexing into its destination slice. The key change is in the scheduler's fusion logic: model `aten.cat` as a "destination-layout epilogue" that assigns each producer a fixed channel interval.

**Affected repro count**: This pattern (multi-branch BN+activation -> cat) appears frequently in Inception-family architectures (timm_adv_inception_v3, timm_inception_v3, etc.).
