# var_mean_9e94237b5daf

## Classification: BN_TRAINING_FUSED_ACTIVATION

## Current Result

- Family: `bn_training_hardswish`
- Oracle path: `repros/canonical/var_mean_9e94237b5daf/oracle_bn_training_hardswish.py`
- Correctness: PASS
- Oracle: `63.23 us`
- `torch.compile coordinate_descent_tuning=True`: `79.68 us`
- Ratio: 1.26
- Best config: `combo+mk=3`: `85.89 us` (WORSE)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete MobileNetV3 training-BatchNorm hard-swish scope for [512, 960, 7, 7] in one per-channel Triton schedule, including the population var_mean, rsqrt(var + 1e-5), mutable running-stat copy_ updates, affine scale/bias, and full [512, 960, 7, 7] hard-swish output.

Inductor lowers the canonicalized training-normalization graph through a generic stats/update kernel plus a separate activation-output schedule.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 79.68 |
| combo+mk=2 | 91.28 (WORSE) |
| combo+mk=3 | 85.89 (WORSE) |
| Oracle | 63.23 |

Multi-kernel configs are worse -- same pattern as var_mean_65e90900fd65 (BN training).

## Root cause

Same root cause as var_mean_65e90900fd65: the norm-template scheduler does not keep all BN side outputs, mutable running-stat epilogues, affine, and fused activation (hard-swish) in one compact per-channel schedule. The training BN template splits the work.

## Kernel count
- Oracle: 1 kernel (per-channel BN + affine + hardswish + running stats)
- Inductor: 2+ kernels (BN stats, then affine+hardswish+running stats)

## Recommendation

Fix in `torch/_inductor/scheduler.py` or norm template: same fix as var_mean_65e90900fd65. Extend the training BatchNorm template so its channel-statistics kernel can emit fused affine activation outputs while preserving the returned running-stat aliases.
