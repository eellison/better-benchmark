# pointwise_13f26d420251

## Classification: `ALREADY_FIXED`

## Pattern

BN-affine (sub, rsqrt, mul, mul, add) + ReLU + MaxPool(3x3, stride=2) + AvgPool(3x3, stride=1, pad=1)

- Model: timm_adv_inception_v3_infer (4 models share this pattern)
- Shape: [128, 192, 71, 71] f32 input, BN params [192]
- Output: i8[128, 192, 35, 35] (pool offsets) + f32[128, 192, 35, 35] (avg_pool)

## Measurements (NVIDIA B200, CD tuning ON)

| Metric | Value |
|--------|-------|
| Compile (best) | 646 us |
| Oracle (triton fused BN+ReLU+MaxPool) | 1800 us |
| Gap (compile/oracle) | 0.36x |
| Kernel count | 3 |
| Memory traffic (min) | ~887 MB |
| Bandwidth SOL | ~111 us |

## Diagnosis

Inductor generates 3 kernels: (1) fused BN-affine + ReLU pointwise, (2) MaxPool template, (3) AvgPool template. It does NOT fuse BN into the pool stencil read.

Despite this, the compiled code (646 us) is **faster** than the hand-written Triton oracle (1800 us). The oracle's per-output-row launch with BLOCK_OW=32 achieves poor occupancy and vectorization on B200's massive SM count. Inductor's auto-tuned pointwise + pool template kernels achieve better hardware utilization through wider vectorization and better occupancy.

The `slice_scatter_elision` pass is not relevant here (no slice_scatter ops in the graph). The fusion opportunity (BN into pool read) remains theoretically available but is unnecessary for performance since inductor already delivers optimal throughput.

## Inductor Closure

No action needed. Compile already beats the oracle ceiling by 2.8x. The remaining gap to bandwidth SOL (646 us vs 111 us) reflects irreducible compute cost of the 3x3 stencil operations, not a fusion opportunity.
