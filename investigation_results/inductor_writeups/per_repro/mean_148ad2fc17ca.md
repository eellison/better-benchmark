# mean_148ad2fc17ca

## Summary
- **Model**: torchbench_mnasnet1_0 (inference)
- **Pattern**: BN-inference (sub, sqrt, reciprocal, mul, add) -> fp16 cast -> ReLU -> spatial mean [256,1280,7,7] -> [256,1280]
- **Previous Ratio**: 1.329x (oracle 22.14us vs compile 29.44us)
- **Current Ratio**: 0.72x (Inductor BEATS oracle: compile 15.9us vs oracle 22.08us)
- **Classification**: ALGEBRAIC_ELIMINATION (FIXED)
- **Status**: BAD_ORACLE (Inductor faster than oracle)

## Fix Applied

Covered by `bn_affine_folding` pass (config `fold_bn_affine = True`).
The pass eliminates per-element sqrt/reciprocal recomputation by precomputing
per-channel scale/shift, then the scheduler fuses the remaining FMA + ReLU + spatial
mean into a single efficient persistent reduction kernel.

## Kernel Count
- **Inductor (after fix)**: 1 kernel (all fused with precomputed BN coefficients)
- **Oracle**: 2 kernels (affine precompute + tiled BN+ReLU+mean)

## Historical Root Cause

Inductor previously recomputed BN algebra (sqrt, reciprocal, mul) for every
(batch, channel) row - 327,680 recomputations instead of 1,280.

Key inefficiencies (now fixed):
1. **Redundant sqrt/reciprocal**: computed 327,680 times instead of 1,280 times
2. **mul by 1.0**: `reciprocal * 1.0` is a no-op but costs an instruction per element

With BN affine folding, per-channel ops are precomputed and the scheduler tiles
efficiently, resulting in Inductor being 28% FASTER than the oracle.

## Config Exploration
- `combo_kernels=True` + `coordinate_descent_tuning=True` already enabled
- BN affine folding closes the gap entirely

## Affected Repro Count
Many BN-inference + spatial mean repros share this pattern. Other affected repros include mean_cf12f53df7dd (MobileNetV3, also BAD_ORACLE now), and all `bn_relu_spatial_mean` and `bn_silu_spatial_mean` variants in the corpus.
