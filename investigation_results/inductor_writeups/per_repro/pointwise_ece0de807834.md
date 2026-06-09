# pointwise_ece0de807834

## Compile: 40.51us, Oracle: 39.65us, Gap: 1.022x

## Diagnosis: AT_FLOOR

## Root cause: The oracle (BN hardswish pointwise) matches torch.compile output within noise. The 2.2% gap is within measurement variance.

## Status: closed_at_floor

## Details

- Model: batch norm + hardswish pointwise
- Pattern: batch norm affine -> hardswish activation (pointwise) [512, 960, 7, 7] fp32
- Shape: [512, 960, 7, 7] fp32 with stride (47040, 49, 7, 1), output contains NaN values (nan_count=11841536)
- Inductor generates an efficient fused kernel essentially matching oracle performance
- Output verified correct (max_finite_diff=1.53e-05, nan_mask_match=True)
- 2.2% gap is within autotuning/measurement variance
- No config exploration needed -- already at floor
