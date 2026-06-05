# pointwise_08c0fb1a5e37 - Layout Transpose

## Benchmark Result
- Oracle: 7.87 us
- Compile: 8.0 us
- Ratio: 1.016x
- Status: AT_FLOOR

## Root Cause
Simple layout transpose operation (stride=[1, 768] output for [768, 2048] tensor). Inductor handles this with a standard pointwise copy kernel that respects the target stride. The ~1.6% difference is within measurement noise.

## Kernel Count
- Oracle: 1 kernel (copy with layout)
- Inductor: 1 kernel (pointwise copy)

## Config Exploration
No config changes needed - already at floor.

## Conclusion
AT_FLOOR - no action needed. Inductor matches oracle for this simple layout operation.
