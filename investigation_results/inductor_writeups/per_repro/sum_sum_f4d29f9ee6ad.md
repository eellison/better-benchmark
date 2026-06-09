# sum_sum_f4d29f9ee6ad

## Compile: 1001.25us, Oracle: 958.43us, Gap: 1.045x

## Classification: AT_FLOOR

## Root Cause

The oracle (softmax_backward_column_sum) is within 4.5% of the compiled output. Both oracle and Inductor are bandwidth-bound on the large [20005, 16384] f32 activation; the 4.5% difference is within measurement noise and does not represent a recoverable scheduling gap.

## Status: at_floor

## Details
- Oracle type: softmax_backward_column_sum
- Shape: [20005, 16384] f32 + [20005] f32 output
- The oracle matches Inductor within noise; no change needed
- multi_kernel and use_fast_math not applicable (gap is below 1.05 threshold)
