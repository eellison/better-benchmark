# var_mean_41a46a395e3a

## Compile: 20.38us, Oracle: 17.31us, Gap: 1.18x

## Diagnosis: NEW_PATTERN

## Root cause

The oracle computes the complete MobileViT patch-layout rewrite and fp32 hidden-dim LayerNorm in one row-parallel Triton kernel. It maps each output row directly to the original NCHW logical indices using the MobileViT patch-unfold indexing (view [N*C*H/2, 2, W/2, 2] -> permute [0,2,1,3] -> view [N, C, patches, 4] -> permute [0,3,2,1] -> view [N*4, patches, C]) and performs the reduction plus affine epilogue without materializing intermediate buffers.

Inductor generates 3 kernels:
1. A partial reduction kernel (`triton_red_fused_clone_permute_var_mean_view_0`) that handles the first pass of the multi-step reduction
2. A per-block finalization kernel (`triton_per_fused_clone_permute_var_mean_view_1`) that computes final mean/variance
3. A pointwise kernel (`triton_poi_fused_add_clone_mul_permute_rsqrt_sub_var_mean_view_2`) that applies normalization

The 3-kernel approach comes from the reduction dimension size (240) being large enough to trigger split-reduction but too small for the oracle's single-pass approach. The oracle handles this in one pass by processing 4 rows at once (ROW_BLOCK=4) with BLOCK_H=256 covering the hidden dimension.

## Kernel count

- Inductor: 3 kernels (partial reduce + finalize + pointwise epilogue)
- Oracle: 1 kernel (fused row-norm with patch indexing)

## Config exploration

| Config | Kernels | Notes |
|--------|---------|-------|
| combo_kernels=True, coord_descent=True | 3 | No improvement |
| combo_kernels=True, combo_kernel_per_subkernel_blocks=True, multi_kernel=1 | 5 | Worse (multi_kernel adds variants) |

## File/line references

- `/tmp/pytorch-work/torch/_inductor/choices.py` - reduction strategy selection (persistent vs split)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - Triton code emission
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion decisions for clone/permute/view chains

## Design doc

The gap (1.18x) is moderate. The root cause is twofold:

1. **Clone/permute/view chain materialization**: The sequence of clone -> view -> permute -> clone -> view -> permute -> clone -> view involves multiple contiguous() calls that Inductor must materialize as intermediate buffers before the reduction can proceed. The oracle bypasses all these by computing the final output row's source coordinates directly from the NCHW input.

2. **Split reduction for dim=240**: Inductor splits the reduction into two passes because 240 exceeds its persistent-reduction heuristic for the given number of rows (8192). The oracle uses BLOCK_H=256 (next power of 2 above 240) in a single persistent pass.

The fix would be a new pattern in `torch/_inductor/fx_passes/` that recognizes the MobileViT patch-unfold sequence (view-permute-clone-view-permute-clone-view followed by layer_norm/var_mean over the last dim) and fuses the index computation into the reduction kernel. However, this is a very specific architectural pattern and the gap is only 1.18x, making it lower priority.

Alternatively, a more general fix would be to improve Inductor's ability to fold view/permute/clone chains into the indexing expression of a downstream reduction, avoiding the intermediate materializations. This is related to the scheduler's handling of `realize_hint()` on contiguous clones.

Model: timm_mobilevit_s inference
Pattern: conv [128,240,8,8] -> clone/view/permute/clone/view/permute/clone/view -> var_mean(dim=2) -> layer_norm affine -> view [8192,240]
Input: [128, 240, 8, 8] NCHW
Output: [8192, 240] (row-normalized)

## Status: design_todo
