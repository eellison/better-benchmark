# pointwise_8793407401ab

## Classification: `ALREADY_FIXED`

## Pattern

BN-affine + ReLU + bilinear interpolation (4x _unsafe_index + lerp) + constant_pad + cat with skip connection

- Model: torchbench_pytorch_unet_infer
- Shape: conv [8, 64, 320, 479] f32, skip [8, 64, 640, 959] f32
- Output: f32[8, 128, 640, 959] (cat of upsampled BN+ReLU with skip)

## Measurements (NVIDIA B200, CD tuning ON)

| Metric | Value |
|--------|-------|
| Compile (best) | 1678 us |
| Oracle (triton fused BN+ReLU+bilinear+cat) | 3074 us |
| Gap (compile/oracle) | 0.55x |
| Kernel count | 3 |
| Memory traffic (min) | ~4085 MB |
| Bandwidth SOL | ~511 us |

## Diagnosis

Inductor generates 3 kernels for this complex pattern. The compile (1678 us) beats the oracle (3074 us) by 1.8x on B200. The oracle attempts to fuse BN+ReLU into the bilinear interpolation and writes directly into a pre-allocated cat buffer, but its kernel launch configuration (per-element programs) achieves poor hardware utilization on B200's massive parallelism.

This is a very large tensor operation (4 GB traffic). The compile achieves 2.4 TB/s effective bandwidth, which is reasonable for a compute-mixed workload (bilinear requires 4 gather loads per output element with lerp arithmetic).

The `slice_scatter_elision` pass is not relevant. The pattern involves `_unsafe_index` (gather-style indexing) rather than slice_scatter.

## Inductor Closure

No action needed. Compile beats oracle on B200. The remaining gap to bandwidth SOL (1678 us vs 511 us) is dominated by the bilinear interpolation's random-access gather pattern, which is inherently bandwidth-inefficient due to non-sequential memory access.
