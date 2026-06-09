# sum_sum_sum_9e7a546b859f

## Compile: 34.94us, Oracle: 35.74us, Gap: 0.978x

## Classification: AT_FLOOR

## Root Cause

The oracle (mobilebert_full_multi_output) is within 2.2% of the compiled output (and actually slightly slower). Both oracle and Inductor achieve near-identical performance on this small MobileBERT backward fragment with [128] f32 and [128, 32768] f32 outputs. At 35us total, the kernel is already at the launch-overhead floor.

## Status: at_floor

## Details
- Oracle type: mobilebert_full_multi_output
- Shapes: multiple [128] f32 vectors + [128, 32768] f32 matrix
- 8 outputs total
- Both implementations are at the hardware floor; no scheduling improvement possible
