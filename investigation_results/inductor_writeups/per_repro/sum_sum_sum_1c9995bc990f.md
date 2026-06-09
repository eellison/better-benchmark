# sum_sum_sum_1c9995bc990f

## Compile: 69.41us, Oracle: 66.53us, Gap: 1.043x

## Classification: AT_FLOOR

## Root Cause

The oracle (cooperative_split_k) is within 4.3% of the compiled output. This is within normal autotuning variance.

## Status: at_floor

## Details
- Oracle type: cooperative_split_k
- Shape: [768, 16384] f32 multi-output reduction
- The 4.3% gap is within measurement noise / autotuning variance
- No Inductor change needed
