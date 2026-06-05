# any_amax_sum_a476543fe91a

## Compile: 11.81us, Oracle: 11.46us, Gap: 1.031x

## Diagnosis: AT_FLOOR

## Root cause: The oracle fuses a full masked attention softmax (with `any` mask check for all-masked rows) into a single Triton kernel. Inductor already achieves essentially the same performance -- the 3.1% gap is within measurement noise for a kernel at this timescale (~12us).

## Status: closed_at_floor

## Details

- Model: full masked softmax (attention pattern with boolean mask)
- Pattern: any (mask check) -> amax -> sub -> exp -> sum -> div (softmax with masked rows)
- Shape: [16, 512, 512] attention scores (fp32)
- Inductor generates a single fused kernel that matches oracle
- 3.1% gap is within autotuning/measurement variance
- No config exploration needed -- already at floor
