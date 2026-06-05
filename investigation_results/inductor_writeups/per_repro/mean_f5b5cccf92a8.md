# mean_f5b5cccf92a8

## Classification: ALGEBRAIC_ELIMINATION

## Current Result

- Oracle path: `repros/canonical/mean_f5b5cccf92a8/oracle_bn_relu_spatial_mean.py`
- Correctness: PASS
- Oracle: 20.22 us
- Compile (cd=True): 21.86 us
- Ratio: 1.081
- Best config: mk=3 combo: 21.38 us (ratio 1.057)
- Status: real_gap (marginal)

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 21.86 | 1.081 |
| combo+mk=2 | 21.79 | 1.076 |
| combo+mk=3 | 21.38 | 1.057 |
| Oracle | 20.22 | 1.000 |

## Root Cause

The oracle computes fp16 BN-inference affine, explicit fp16 rounding, NaN-preserving ReLU, and 28x28 spatial mean directly into the final fp16 [256,C,1,1] keepdim output, while hoisting per-channel scale/shift. It algebraically eliminates loads for rows whose runtime variance+eps makes the affine result NaN for every spatial element.

Inductor lowers the broadcast normalization, activation, cast, and spatial reduction through generic norm-template reduction code that still reads the full activation tile for NaN-dominated rows.

## Kernel count
- Oracle: 2 kernels (affine precompute + fused spatial mean)
- Inductor: 2+ kernels (norm template + reduction)

## Recommendation

The gap is marginal (< 1.1x even at worst). The ALGEBRAIC_ELIMINATION of NaN-dominated rows is a highly specialized optimization that requires value-dependent guarding. Not cost-effective to pursue for this small gap. Multi_kernel=3 nearly closes it.
