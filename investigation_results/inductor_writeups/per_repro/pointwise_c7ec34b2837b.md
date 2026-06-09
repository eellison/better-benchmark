# pointwise_c7ec34b2837b — relu_flatten

## Status: AT_FLOOR (ratio 0.982x)

## Summary
The compiled output matches or beats the oracle performance. The oracle performs
relu + flatten on shape [512, 1280] float32 tensors.

## Benchmark Results
- Oracle: 7.04 us
- Compiled: 6.91 us
- Ratio: 0.982x

## Conclusion
No optimization needed. Inductor already generates code that matches or exceeds the
oracle's performance.
