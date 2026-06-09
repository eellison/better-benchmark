# pointwise_6d6d27100ad9 — oracle_bn_relu

## Summary
Oracle: 6.46us | Compiled: 6.75us | Ratio: 1.045x | Status: AT_FLOOR

## Classification: NO_GAP

The compiled code is within noise floor of the oracle (ratio < 1.05x). No investigation needed.

## Config exploration
- Default (coordinate_descent_tuning=True, combo_kernels=True): 6.75us
- Ratio 1.045x is within measurement noise

## Root cause
N/A - no meaningful performance gap. The oracle computes batch normalization + ReLU for a [128, 168, 4, 4] tensor. Inductor already generates efficient code for this small tensor size, as the operation is memory-bandwidth bound and near the hardware floor.

## Kernel count
- Inductor: 1 kernel (fused pointwise)
- Oracle: 1 kernel

## Conclusion
No action needed. Inductor matches oracle performance.
