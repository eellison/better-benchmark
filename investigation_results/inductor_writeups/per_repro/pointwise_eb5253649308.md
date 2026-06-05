# pointwise_eb5253649308 — Full Fill

## Status: AT_FLOOR (ratio = 1.029x)

## Oracle Description
The oracle allocates a fresh contiguous cuda float32[8,1,256,257] tensor and fills it with -inf using one Triton store kernel. This is a pure memory-write operation with no input reads.

## Classification: BANDWIDTH_BOUND

## Benchmark Results
- Oracle: 5.44 us
- Compiled: 5.6 us
- Ratio: 1.029x

## Root Cause
This is `aten.full` — a zero-input, one-output constant fill operation. The work is entirely at the allocation + write bandwidth floor. Inductor already compiles this to a single constant-store kernel. The 2.9% difference is within measurement noise.

## Kernel Count
- Oracle: 1 kernel (fill with -inf)
- Inductor: 1 kernel (fill with -inf)

## Config Exploration
No config changes needed — already at bandwidth floor.

## Conclusion
No action required. This is at the absolute write-bandwidth floor. No optimization is possible within this isolated scope.
