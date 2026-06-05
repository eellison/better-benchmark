# pointwise_1383c536f620 — cat + BN + ReLU fusion (DenseNet)

## Summary
- **Repro**: `torchbench_densenet121_infer_000`
- **Ratio**: 1.229 (oracle 15.36us vs compile 18.88us)
- **Classification**: CODEGEN_TILING

## What the oracle does
The oracle fuses cat(8 inputs along channel dim) -> BN inference affine -> fp16 cast -> ReLU into one kernel with a 2D tiling strategy: BLOCK_ROWS=16 (channel rows) x BLOCK_HW=64 (spatial elements). This ensures BN parameters (mean, var, weight, bias) are loaded once per channel-row block and reused across all HW=49 spatial positions in that tile.

The oracle uses `tl.where` chains on pointer values to implement a virtual cat (selecting from 8 different source tensors based on channel index), then applies BN normalization and ReLU.

## What Inductor does
Inductor already fuses this into a single kernel (`triton_poi_fused_add_cat_convert_element_type_mul_reciprocal_relu_sqrt_sub_unsqueeze_0`) with virtual cat using masked loads and `tl.where` chains. The fusion is correct and complete.

**Key difference**: Inductor uses a 1D flat tiling (Grid1D, xnumel=2308096) where each thread processes one element. The BN parameter loads use `evict_last` but because the iteration is flat over (N,C,H,W), consecutive threads within a block access the same channel for only HW=49 elements before moving to the next channel. This means BN params are reloaded more frequently than in the oracle's explicit 2D tiling.

## Kernel count
- Inductor: 1 kernel
- Oracle: 1 kernel

## Config exploration
Inductor with `coordinate_descent_tuning=True` and `combo_kernels=True` already achieves the best fusion strategy. The gap is purely in the tiling/scheduling of the single kernel, not in fusion decisions.

## Root cause
The 1D pointwise tiling strategy does not exploit the broadcast structure of per-channel parameters. The oracle's 2D tiling (channels x spatial) ensures BN parameter loads happen once per channel-block and are reused across all spatial positions in that tile, reducing L2 traffic for the 4 x 736-element parameter vectors.

## Fix path
**Design doc**: The Inductor pointwise codegen could benefit from a "broadcast-aware tiling" heuristic that detects per-channel broadcast parameters (shape [C] broadcast into [N,C,H,W]) and switches to a 2D grid where the inner dimension covers the spatial (HW) extent and the outer covers (N*C) rows. This would be in:
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (tiling selection)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (broadcast detection for tiling hints)

This is a general pattern affecting all BN-like operations fused with upstream producers. The existing `tiling_scores` mechanism would need to be extended to consider broadcast parameter reuse as a tiling preference signal.
