# sum_sum_sum_2005ccbce613

## Compile: 64.16us, Oracle: 63.33us, Gap: 1.013x

## Classification: AT_FLOOR

## Root Cause

The oracle (cooperative_split_k) is within 1.3% of the compiled output. Inductor already matches oracle performance.

## Status: at_floor

## Details
- Oracle type: cooperative_split_k
- Shape: [1024, 8192] f32 multi-output reduction
- The 1.3% gap is within measurement noise
- No Inductor change needed
