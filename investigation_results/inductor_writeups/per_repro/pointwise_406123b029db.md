# pointwise_406123b029db — AT_FLOOR (ratio 1.036x)

## Summary
Oracle: `oracle_avgpool_bn_relu.py` — fused avg_pool2d + BN-inference affine + ReLU
Benchmark: oracle=6.24us, compile=6.46us, ratio=1.036x

## Classification: SCHEDULER_FUSION (at floor)

## Root Cause
The oracle fuses a fixed-shape 2x2 stride-2 avg_pool2d, BN-inference affine, and ReLU
into one Triton output-tiled kernel. The diagnosis indicates Inductor schedules the
structured pooling producer and downstream per-channel affine pointwise as separate regions.

However, at this shape (likely small spatial after pooling), the gap is only 3.6% which
is within noise / at the performance floor. The overhead of separate kernel launches is
negligible at this scale.

## Kernel Count
- Inductor: likely 2 kernels (pool + BN+ReLU)
- Oracle: 1 kernel

## Config Exploration
No config changes needed — ratio below investigation threshold.

## Conclusion
The fusion gap exists in principle (pool + BN + ReLU not fused) but the performance
impact is negligible at this shape. The stochastic nature of the outputs (noted during
correctness check) suggests this may involve dropout-like operations.
