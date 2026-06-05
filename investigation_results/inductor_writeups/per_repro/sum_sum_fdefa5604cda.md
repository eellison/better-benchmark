# sum_sum_fdefa5604cda

## Compile: 75.62us, Oracle: 87.81us, Gap: 0.861x

## Classification: BAD_ORACLE

## Root Cause

The oracle (cooperative_split_k) is slower than the compiled output by 14%. The compiled code already handles this workload efficiently. The oracle's cooperative split-K approach adds overhead that exceeds the theoretical benefit for this particular shape.

## Status: bad_oracle

## Details
- Oracle type: cooperative_split_k
- Shape: [512, 8192] f32 reduction
- The compiled persistent/looped reduction is already optimal for this reduction shape
- No Inductor change needed
