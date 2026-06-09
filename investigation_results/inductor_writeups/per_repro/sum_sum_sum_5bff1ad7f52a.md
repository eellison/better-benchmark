# sum_sum_sum_5bff1ad7f52a

## Compile: 380.77us, Oracle: 376.9us, Gap: 1.010x

## Classification: AT_FLOOR

## Root Cause

The oracle (layernorm_bwd_cooperative) is within 1% of the compiled output. Both oracle and Inductor are already near-optimal for this layernorm backward with outputs [512] f32 and [1152000, 512] bf16. The 1% difference is well within measurement noise.

## Status: at_floor

## Details
- Oracle type: layernorm_bwd_cooperative
- Shapes: [512] f32 + [1152000, 512] bf16 output
- Inductor already matches oracle performance; no change needed
