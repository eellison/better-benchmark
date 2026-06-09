# amax_sum_any_fd11cc31d3d3

## Status: NO GAP (BAD_ORACLE)

## Measurement
- Oracle: 97.15 us
- Compiled: 75.87 us
- Ratio: 0.781x (oracle is slower than compiled)

## Pattern
MobileBERT attention softmax + dropout. Pattern hash `fd11cc31d3d3`, from `hf_MobileBertForMaskedLM_train_000`.
Shape: `[1024, 128, 128]` (batch=256, heads=4, seq=128).

## What the Oracle Does
Custom Triton kernel that fuses:
- Always-true iota/ge broadcast mask folding to zero
- [1024,128,128] to [256,4,128,128] view
- Stable last-dimension softmax with all-minus-inf row zeroing
- Exact Inductor RNG dropout and scale
- Final [1024,128,128] transpose view

## Root Cause of BAD_ORACLE
Inductor's compiled output already handles this efficiently. At shape [256, 4, 128, 128]
with K=128, rows are small enough for Inductor's persistent reduction to handle the
softmax in a single pass with good occupancy. The oracle's approach of fusing dropout
into the same kernel adds register pressure and doesn't save enough memory traffic to
compensate at this shape. Inductor wins by 22%.

## Classification
NEW_PATTERN (per oracle docstring), but no performance gap exists. Inductor already
exceeds oracle performance. No fix needed.

## Kernel Count
Not investigated (no gap to close).
