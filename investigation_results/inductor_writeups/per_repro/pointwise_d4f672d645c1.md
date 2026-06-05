# pointwise_d4f672d645c1

## Compile: 106.46us, Oracle: 74.59us, Gap: 1.43x

## Diagnosis: SCHEDULER_FUSION

## Root cause

The oracle computes the full sigmoid-gated residual SiLU scaling and 2x2 stride-2 avg_pool2d in one Triton output-tiled stencil kernel that reads the unfused inputs ([128,1536,1,1] gate, [128,1536,14,14] conv, [128,1536,14,14] residual) and writes only the final f32 [128,1536,7,7] tensor. Inductor materializes the full f32 [128,1536,14,14] pointwise result (38.5M elements, ~147MB) in one kernel and then runs a separate avg_pool2d kernel over that intermediate buffer.

The fundamental issue is that `aten.avg_pool2d` is in the `realize_users` list in `torch/_inductor/lowering.py` (line 242), which forces its input to be materialized. This prevents the scheduler from fusing the pointwise producer into the pooling consumer.

## Kernel count

- Inductor: 2 kernels (pointwise + avg_pool2d)
- Oracle: 1 kernel (fused stencil)

## Config exploration

| Config | Kernels | Notes |
|--------|---------|-------|
| combo_kernels=True, coord_descent=True | 2 | No improvement |
| combo_kernels=True, combo_kernel_per_subkernel_blocks=True, multi_kernel=1 | 2 | No improvement |

The config space cannot help here because the fusion barrier is structural: avg_pool2d is a fallback op that forces input realization.

## File/line references

- `/tmp/pytorch-work/torch/_inductor/lowering.py:242` - avg_pool2d in realize_users list
- `/tmp/pytorch-work/torch/_inductor/ir.py:10144` - should_realize_on_reuse logic

## Design doc

The fix requires teaching Inductor's scheduler/codegen to fuse broadcast pointwise producers through fixed-window avg_pool2d consumers. This means:

1. Recognizing that avg_pool2d with fixed kernel/stride can be expressed as a stencil (output-tiled loop that reads 2x2 windows from the producer)
2. Instead of materializing the pointwise output, compute the pointwise values on-the-fly for each window position within the stencil kernel
3. This saves the 147MB intermediate buffer round-trip (write 38.5M f32 + read 38.5M f32)

The pattern is common in EfficientNet/NFNet architectures where activation functions are followed by stride-2 downsampling pools. The affected memory traffic is significant: the intermediate buffer is 4x larger than the final output.

Model: timm_nfnet_l0 inference
Pattern: sigmoid(gate) * conv * 2 * 0.2 + residual -> SiLU -> avg_pool2d([2,2], stride=[2,2])

## Status: design_todo
