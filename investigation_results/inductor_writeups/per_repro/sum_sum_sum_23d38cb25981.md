# sum_sum_sum_23d38cb25981

## Compile: 232.42us, Oracle: 430.11us, Gap: 0.540x

## Classification: BAD_ORACLE

## Root Cause

The oracle (cooperative_split_k) is significantly slower than the compiled output (1.85x slower). The cooperative split-K approach is counterproductive for this workload shape.

## Status: bad_oracle

## Details
- Oracle type: cooperative_split_k
- Shape: [128, 401408] f32 multi-output reduction
- The compiled looped reduction handles this very large rnumel (401408) more efficiently than the cooperative approach
- The oracle's coordination overhead exceeds its parallelism benefit at this extreme reduction size
- No Inductor change needed
