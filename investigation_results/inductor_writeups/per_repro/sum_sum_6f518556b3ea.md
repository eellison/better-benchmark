# sum_sum_6f518556b3ea Investigation

## Summary
- **Model**: timm_mobilenetv2_100_train_001
- **Ratio**: 1.512x (oracle: 25.5us, compile: 38.56us)
- **Classification**: SCATTER_REDUCE

## Root Cause

The oracle computes the MobileNetV2 adaptive-average-pool backward, hard-ReLU6 derivative, and batch-norm-backward return tuple directly from the original `[128, 1280]` pooled gradient and `[128, 1280, 7, 7]` activation. It avoids materializing the zero-fill `as_strided_scatter -> as_strided -> expand -> div` pool-gradient tensor by recognizing the structured pattern and computing the pool backward inline (div by 49).

Inductor lowers the structured scatter/expand producer as ordinary tensor work and schedules the hard-ReLU6 mask, two sibling channel reductions, and dependent full pointwise BN-backward epilogue as separate generic consumers.

Shape: [128, 1280, 7, 7], reduction over [0, 2, 3] = 6272 elements per channel, 1280 channels.

## Kernel Count
- **Oracle**: Fused structured scatter-reduce kernel(s)
- **Inductor**: Multiple kernels (scatter materialization + reduction + epilogue)

## Config Exploration
combo_kernels and coordinate_descent_tuning do not help because the fundamental issue is that Inductor materializes an unnecessary intermediate buffer (the zero-fill scatter result) and then reads it back.

## Key Files
- `/tmp/pytorch-work/torch/_inductor/fx_passes/` - structured scatter lowering
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion across scatter boundaries

## Design Doc

The fix requires a structured average-pool-backward scatter/expand lowering that:
1. Recognizes the `full(0) -> as_strided_scatter -> as_strided -> expand -> div` pattern as avg_pool2d_backward
2. Maps pooled-gradient source elements directly into gated channel reductions
3. Emits the dependent BN-backward output tensor in one fused template
4. Avoids materializing the full [128, 1280, 7, 7] intermediate scatter result

This is an FX pass problem: the pass should canonicalize the scatter/expand chain back into a recognized pattern that the scheduler can fuse efficiently.
