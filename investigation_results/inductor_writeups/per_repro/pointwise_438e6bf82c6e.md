# pointwise_438e6bf82c6e

## Compile: 15.04us, Oracle: 14.88us, Gap: 1.011x

## Classification: AT_FLOOR

## Root Cause

The oracle (ReLU + maxpool + flatten) is within 1.1% of the compiled output. Both are at the performance floor.

## Kernel Count
- Oracle and Inductor produce equivalent performance

## Config Exploration
| Config | Result |
|--------|--------|
| combo_kernels + CDT | 15.04 us (1.011x) |

## Status: at_floor

## Details
- Model: ReLU + maxpool + flatten
- Shape: int8 + [128, 25088] f16 outputs
- The 1.1% gap is pure measurement noise
- No Inductor change needed
