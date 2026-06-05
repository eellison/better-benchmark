# var_mean_dc2d388af48f

## Classification: BN_TRAINING_ACTIVATION_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_dc2d388af48f/oracle_bn_relu6_training.py`
- Correctness: PASS
- Oracle: 17.89 us
- Compile (cd=True): 22.43 us
- Ratio: 1.254
- Best config: combo+mk=3: 22.46 us (ratio 1.232)
- Status: real_gap

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 22.43 | 1.254 |
| combo+mk=2 | 22.43 | 1.234 |
| combo+mk=3 | 22.46 | 1.232 |
| Oracle | 17.89 | 1.000 |

No config closes the gap.

## Root Cause

The oracle computes the complete MobileNetV2 training-BatchNorm ReLU6 scope for [128, 960, 7, 7] in one per-channel Triton schedule, including:
- Population var_mean over (N, H, W) per channel
- rsqrt(var + 1e-5)
- Mutable running-stat copy_ updates (running_mean, running_var)
- Affine scale/bias
- ReLU6 clamp
- Full [128, 960, 7, 7] output

Inductor lowers the canonicalized training-normalization graph through generic normalization/update scheduling instead of one full-scope BN-plus-activation template. The norm-template scheduler does not keep the affine ReLU6 materialization and returned running-stat aliases inside one reusable training-BN reduction plan.

## Kernel count
- Oracle: 1 kernel (fused BN stats + affine + ReLU6 + running stat updates)
- Inductor: 2+ kernels (BN statistics + affine/activation + running stat updates)

## Recommendation

Extend the training BatchNorm template so its channel-statistics kernel can emit fused affine activation outputs while preserving mutable running-stat returns.

File: `torch/_inductor/scheduler.py` (norm template fusion with activation epilogue)
File: `torch/_inductor/codegen/triton.py` (BN training template)
