# pointwise_b673b7143973

## Compile: 10.98us, Oracle: 10.94us, Gap: 1.003x

## Diagnosis: AT_FLOOR

## Root cause: The oracle (rowwise affine) matches torch.compile output within noise. The 0.3% gap is not meaningful.

## Status: closed_at_floor

## Details

- Model: rowwise affine transformation
- Pattern: pointwise affine (mul + add with per-row weights) [32768, 128] fp32
- Shape: [32768, 128] fp32 single output
- Inductor generates an efficient fused kernel matching oracle performance
- Output verified correct (max_diff=9.54e-07)
- No config exploration needed -- already at floor
