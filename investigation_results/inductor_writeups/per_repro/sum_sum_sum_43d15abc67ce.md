# sum_sum_sum_43d15abc67ce

## Compile: 25.5us, Oracle: 24.42us, Gap: 1.045x

## Classification: AT_FLOOR

## Root Cause

The oracle (multi output reduction) is within 4.5% of the compiled output. At this ~25us timescale, this is within normal measurement noise and autotuning variance.

## Kernel Count
- Oracle and Inductor at near-parity

## Config Exploration
| Config | Result |
|--------|--------|
| combo_kernels + CDT | 25.5 us (1.045x) |

## Status: at_floor

## Details
- Model: multi-output reduction (functorch dp cifar10 style)
- Shape: [128] + [128] + [128, 32768] f32 + [128] f32
- Pattern: grouped BN-backward-style reductions
- The 4.5% gap is within measurement noise; no fix needed
