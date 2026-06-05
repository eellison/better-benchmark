# pointwise_4f6d85385d17

## Compile: 82.66us, Oracle: 82.56us, Gap: 1.001x

## Classification: AT_FLOOR

## Root Cause

The oracle (virtual cat avgpool) is essentially identical in performance to the compiled output. The 0.1% difference is pure measurement noise.

## Kernel Count
- Oracle and Inductor produce equivalent performance

## Status: at_floor

## Details
- Model: virtual cat avgpool
- Shape: int8 + stochastic output
- Pattern: cat + avg_pool fusion
- No gap exists; both are at the performance floor
