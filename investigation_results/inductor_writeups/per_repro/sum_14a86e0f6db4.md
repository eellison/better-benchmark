# sum_14a86e0f6db4

## Compile: 56.22us, Oracle: 54.94us, Gap: 1.023x

## Classification: AT_FLOOR

## Root Cause

The gap is within noise (1.023x < 1.05x threshold). Inductor is effectively matching the oracle performance for this visformer softmax backward scale workload.

## Kernel Count
- Oracle: N/A (at parity)
- Inductor: effectively matching oracle

## Config Exploration
No deep exploration needed -- gap is below 1.05x threshold.

## Fix Assessment: No action needed

Performance is at floor -- Inductor matches the oracle within measurement noise.

## Details
- Model: Visformer softmax backward with scale
- Shape: output [768, 196, 196] f32
