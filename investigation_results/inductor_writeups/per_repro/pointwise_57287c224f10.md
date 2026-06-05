# pointwise_57287c224f10 — ResNet Dual BN Add ReLU

## Summary
- **Repro**: (resnet dual batchnorm + add + relu pattern)
- **Oracle**: oracle_resnet_dual_bn_add_relu.py
- **Ratio**: 1.043x (oracle 19.26us vs compile 20.10us)
- **Status**: AT_FLOOR (within noise margin, below 1.05x threshold)

## Benchmark Result
The ratio of 1.043x is below the 1.05x investigation threshold. While there may be a
small gap, it is within measurement noise and does not warrant deep investigation.
