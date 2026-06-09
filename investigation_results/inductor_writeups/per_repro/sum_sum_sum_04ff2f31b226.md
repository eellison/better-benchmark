# sum_sum_sum_04ff2f31b226

## Compile: 20.10us, Oracle: 20.29us, Gap: 0.991x

## Classification: AT_FLOOR

## Root Cause

The oracle (pool_scatter_reduce) is within 1% of the compiled output. The compiled code already matches oracle performance for this workload.

## Status: at_floor

## Details
- Oracle type: pool_scatter_reduce
- Shape: [64, 64, 16, 16] f32 + [64] f32 channel reductions
- The 1% difference is within measurement noise
- No Inductor change needed
