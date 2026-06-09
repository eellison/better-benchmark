# sum_sum_d550f2d61c6c

## Compile: 331.52us, Oracle: 171.74us, Gap: 1.93x

## Diagnosis: SCHEDULER_FUSION

## Root cause: Inductor schedules the masked MobileNetV2 batchnorm-backward fragment as 4 separate Triton kernels: (1) a pointwise kernel computing the BN affine (sub, mul, unsqueeze, add), (2) a reduction kernel computing the two channel sums (sum_where, sum_centered) over the masked gradient, (3) a per-reduction fixup kernel, and (4) a final pointwise kernel writing the output tensor using the reduction results. The oracle fuses everything into 2 coordinated kernels: one that computes per-batch partial sums per channel (C*N grid = 32*128 = 4096 programs), and one that finalizes the sums and writes the full [128,32,112,112] output in the same kernel launch.

## Kernel count
- Inductor: 4 kernels
- Oracle: 2 kernels (partial sums + write output)

## Config exploration
- `coordinate_descent_tuning=True`: already enabled, produces current 4-kernel schedule
- `combo_kernels=True`: already enabled, doesn't help fuse these across reduction boundary
- The fundamental issue is that Inductor materializes the BN affine intermediate ([128,32,112,112] = 229MB) between kernels, and re-reads the full input tensor multiple times across the 4 kernel launches

## Fix path: Add scheduler/codegen support for "masked BN-backward full-scope" fusion. The key enhancement needed: when the scheduler sees (1) a pointwise producer, (2) two sibling channel reductions consuming the same masked derivative of that producer, and (3) a dependent pointwise epilogue consuming both reductions' results plus the same inputs, it should fuse all into a 2-phase kernel plan:
- Phase 1: per-(channel, batch) partial sums (cheap, parallelizable)
- Phase 2: finalize sums + write full output tensor with epilogue

This avoids materializing the [128,32,112,112] intermediate and reading the input 4 times.

## Status: design_todo

## Details

- Model: timm_mobilenetv2_100 (train)
- Pattern: sub, mul, add (BN affine) -> le/ge/bitwise_or (hardswish/clamp mask) -> where -> sum([0,2,3]) x2 -> dependent BN-backward epilogue
- Shape: [128,32,112,112] f32 (N=128, C=32, H=W=112)
- Reduction size per channel: N*H*W = 128*112*112 = 1,605,632 elements
- Data volume: 128*32*112*112*4 = ~229MB per full tensor
- The 1.93x gap comes from:
  1. Materializing [128,32,112,112] affine intermediate (229MB write + read)
  2. Reading the same large tensors across 4 kernel launches instead of 2
  3. Total extra memory traffic: ~460MB compared to oracle's coordinated 2-kernel plan
- File references:
  - `/tmp/pytorch-work/torch/_inductor/scheduler.py`: fusion scoring doesn't recognize the BN-backward pattern as fusible across the reduction boundary
  - `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: no template for partial-sum + finalize-and-write pattern
  - `/tmp/pytorch-work/torch/_inductor/ir.py`: realize_hint may force materialization of the affine intermediate
