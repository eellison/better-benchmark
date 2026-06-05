# var_mean_65e90900fd65

## Classification: BN_TRAINING_FUSED_ACTIVATION

## Current Result

- Family: `bn_training_relu`
- Oracle path: `repros/canonical/var_mean_65e90900fd65/oracle_bn_training_relu.py`
- Correctness: PASS
- Oracle: `9.86 us`
- `torch.compile coordinate_descent_tuning=True`: `13.02 us`
- Ratio: 1.321
- Best config: `combo+mk=3`: `44.48 us` (WORSE - wrong strategy for BN)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete Inception training-BatchNorm affine ReLU scope in one per-channel Triton program for [128, 384, 8, 8], including population var_mean over [128,8,8], eps=0.001 rsqrt, [384] invstd output, [1,384,1,1] saved-mean output, in-place running mean/variance copy_ updates, and the full [128,384,8,8] ReLU activation.

Inductor lowers the decomposed norm-template graph through its generic training-normalization schedule, which does not keep all BN side outputs, mutable running-stat epilogues, affine, and ReLU in one compact per-channel schedule.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 13.02 |
| combo+mk=2 | 51.96 (WORSE) |
| combo+mk=3 | 44.48 (WORSE) |
| Oracle | 9.86 |

Multi-kernel configs are catastrophically worse -- they change the reduction strategy away from per-channel parallelism which is optimal for BN training.

## Root cause

The norm-template scheduler does not keep all BN side outputs (invstd, saved_mean), mutable running-stat epilogues (copy_ updates), affine, and fused activation (ReLU) in one compact per-channel schedule. The training BN template splits the work across multiple kernels.

## Kernel count
- Oracle: 1 kernel (per-channel BN + affine + ReLU + running stats)
- Inductor: 2+ kernels (BN stats, then affine+ReLU+running stats)

## Recommendation

Fix in `torch/_inductor/scheduler.py` or norm template: extend the training BatchNorm norm template so it can emit the side outputs and fused affine/ReLU epilogue from the channel-statistics kernel when profitable. File: `torch/_inductor/codegen/triton.py`.
