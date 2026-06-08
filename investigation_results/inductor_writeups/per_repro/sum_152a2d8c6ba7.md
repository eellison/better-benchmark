# sum_152a2d8c6ba7

## Compile: 57.02us, Oracle: 56.99us, Gap: 1.001x

## Classification: AT_FLOOR

## Root Cause

The gap is within noise (1.001x < 1.05x threshold). Inductor is effectively matching the oracle performance for this visformer softmax backward scale workload.

## Kernel Count
- Oracle: N/A (at parity)
- Inductor: effectively matching oracle

## Config Exploration
No deep exploration needed -- gap is below 1.05x threshold.

## Fix Assessment: No action needed

Performance is at floor -- Inductor matches the oracle within measurement noise.

## Details
- Model: Visformer softmax backward with scale
- Shape: output [768, 196, 196] f32, stride [38416, 196, 1]
