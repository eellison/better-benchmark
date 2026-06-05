# sum_sum_sum_33f45f532ce3

## Compile: 85.89us, Oracle: 87.81us, Gap: 0.978x (BAD_ORACLE at noise)

## Classification: AT_FLOOR

## Root Cause

Compile is already slightly faster than the oracle. Both are at the performance floor for this structured select_scatter reduce workload.

## Status: at_floor

## Details
- Model: structured select scatter reduce
- Shape: [768] f32 reductions + [768, 25344] f32 side output + [768] f32
- Oracle and compile are at parity (oracle is marginally slower)
- No Inductor change needed
