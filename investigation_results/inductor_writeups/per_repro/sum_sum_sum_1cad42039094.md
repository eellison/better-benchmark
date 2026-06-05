# sum_sum_sum_1cad42039094

## Compile: 128.80us, Oracle: 126.98us, Gap: 1.014x

## Classification: AT_FLOOR

## Root Cause

The oracle (multi_output_reduction) is within 1.4% of the compiled output. Inductor already matches oracle performance.

## Status: at_floor

## Details
- Oracle type: multi_output_reduction
- Shape: [256, 100352] f32 multi-output reduction
- The 1.4% gap is within measurement noise
- No Inductor change needed
