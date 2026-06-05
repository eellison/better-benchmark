# sum_sum_sum_33dcb3bff2b0

## Compile: 80.90us, Oracle: 122.59us, Gap: 0.660x

## Classification: BAD_ORACLE

## Root Cause

The oracle (cooperative_split_k) is significantly slower than the compiled output (1.52x slower). The cooperative approach is counterproductive for this workload.

## Status: bad_oracle

## Details
- Oracle type: cooperative_split_k
- Shape: [768] f32 multi-output reduction with [1, 1, 768] intermediate
- The compiled code handles this efficiently without cooperative splitting overhead
- No Inductor change needed
