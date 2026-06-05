# pointwise_c83a7895e053

## Compile: 8.54us, Oracle: 8.99us, Gap: 0.95x (BAD_ORACLE)

## Classification: NO_GAP

## Root Cause

Compile is already faster than the oracle by 5%. No investigation needed.

## Status: closed_no_gap

## Details
- Model: SiLU activation
- Shape: [128, 384, 7, 7] f32
- Oracle attempts fused SiLU but Inductor already matches or beats it
- No Inductor change needed
