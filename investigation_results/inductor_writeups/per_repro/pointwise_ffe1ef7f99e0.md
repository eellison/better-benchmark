# pointwise_ffe1ef7f99e0 - BN Affine

## Summary
- **Pattern**: BatchNorm affine transformation
- **Shape**: [512, 80, 7, 7] (fp32)
- **Ratio**: 0.93x (oracle 9.12 us, compiled 8.48 us)
- **Status**: BAD_ORACLE - compiled is already faster than the oracle

## Verdict
No action needed. Inductor outperforms the oracle by ~7%. The oracle's hand-written kernel is suboptimal for this BN affine pattern at this shape.
