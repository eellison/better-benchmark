# pointwise_71e3a6c09140

## Classification: `ALREADY_FIXED`

## Pattern

BN-affine (sub, rsqrt, mul, mul, add) + ReLU + MaxPool(3x3, stride=2, no padding, no ceil)

- Model: timm_adv_inception_v3_infer (4 models share this pattern)
- Shape: [128, 64, 147, 147] f32 input, BN params [64]
- Output: f32[128, 64, 73, 73] (pool values) + i8[128, 64, 73, 73] (pool offsets)

## Measurements (NVIDIA B200, CD tuning ON)

| Metric | Value |
|--------|-------|
| Compile (best) | 749 us |
| Oracle (triton fused BN+ReLU+MaxPool) | 1286 us |
| Gap (compile/oracle) | 0.58x |
| Kernel count | 2 |
| Memory traffic (min) | ~926 MB |
| Bandwidth SOL | ~116 us |

## Diagnosis

Inductor generates 2 kernels: (1) fused BN-affine + ReLU + MaxPool (the pool template reads the fused pointwise inline), (2) the pool offsets output or a second pass. The compiled code (749 us) is **faster** than the oracle (1286 us) on B200.

The oracle's Triton kernel uses per-row programs with BLOCK_OW=32, which does not saturate B200's SM count and memory subsystem as efficiently as inductor's auto-tuned template. On B200, the wider bus and higher SM count reward vectorized access patterns that inductor's heuristics generate better than the hand-written kernel.

The `slice_scatter_elision` pass is not relevant here. No additional fusion is needed.

## Inductor Closure

No action needed. Compile already beats the oracle by 1.7x. Inductor's existing pool template with fused pointwise is more efficient than the hand-written alternative on this hardware.
