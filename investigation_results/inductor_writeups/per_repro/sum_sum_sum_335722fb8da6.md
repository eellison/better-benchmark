# sum_sum_sum_335722fb8da6

## Compile: 16.38us, Oracle: 17.54us, Gap: 0.934x

## Classification: BAD_ORACLE

## Root Cause

The oracle (cooperative_split_k) is slower than the compiled output by 7%. The compiled code handles this bf16 workload more efficiently.

## Status: bad_oracle

## Details
- Oracle type: cooperative_split_k
- Shape: [1024, 2048] bf16
- The compiled reduction is already optimal for this moderate reduction size in bf16
- No Inductor change needed
