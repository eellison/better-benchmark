# pointwise_baa1198c62c0 — layout_clone

## Status: AT_FLOOR (ratio 0.992x)

## Summary
The compiled output matches or beats the oracle performance. The oracle performs a
layout clone operation on shape [25216, 192] float32 tensors.

## Benchmark Results
- Oracle: 11.84 us
- Compiled: 11.74 us
- Ratio: 0.992x

## Conclusion
No optimization needed. Inductor already generates code that matches or exceeds the
oracle's performance.
