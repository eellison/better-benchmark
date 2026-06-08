# var_mean_aa8ab7fa55a7

## Classification: SCHEDULER_FUSION

## Current Result

- Family: `swin_patch_merge_layernorm`
- Oracle path: `repros/canonical/var_mean_aa8ab7fa55a7/oracle_*`
- Compiled (coordinate_descent_tuning=True): 36.58 us
- Status: `real_gap`

## Diagnosis

The oracle computes Swin patch-merge residual add, fixed 2x2 layout clone, population var_mean LayerNorm, affine epilogue, and final [6272, 2048] reshape in one row kernel. Inductor schedules the layout materialization and normalization separately.

## Root cause

The normalization scheduler does not recognize the deterministic Swin patch-merge reshape/permute/clone producer as a direct row-source for the LayerNorm template.

## Kernel count
- Oracle: 1 kernel
- Inductor: 1+ kernels

## Recommendation

Teach the norm-template scheduler to sink fixed patch-merge layout indexing and residual-add producers into the row-wise LayerNorm load plan.

## Relevant files

- Input: [25088, 512] + [128, 196, 512] f32 (Swin transformer inference)
- Models: timm_swin_base_patch4_window7_224_infer
