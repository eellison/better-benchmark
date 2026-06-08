# var_mean_a1464365da4f

## Classification: SCHEDULER_FUSION

## Current Result

- Family: `nfnet_channel_normalization`
- Oracle path: `repros/canonical/var_mean_a1464365da4f/oracle_*`
- Compiled (coordinate_descent_tuning=True): 11.97 us
- Status: `real_gap` (marginal)

## Diagnosis

The oracle computes NFNet per-channel population var_mean, rsqrt side output, scalar-gain normalized activation, and keepdim mean side output in one shape-specialized Triton program. Inductor lowers through its generic var_mean normalization schedule without retaining the 1536-element channel tile.

## Root cause

The scheduler lacks a guarded reduction-consumer fusion template for fixed inner-size correction=0 channel normalization with multiple returned reduction consumers.

## Kernel count
- Oracle: 1 kernel
- Inductor: 1 kernel

## Recommendation

Add a per-channel var_mean+rsqrt+scale-normalize template that fuses the dependent broadcast epilogue and emits the requested side-output layouts directly.

## Relevant files

- Input: [3072, 1536, 1, 1] f32 (NFNet training)
- Models: 5 models (timm_dm_nfnet)
