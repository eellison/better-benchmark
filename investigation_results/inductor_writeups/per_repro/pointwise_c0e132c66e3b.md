# pointwise_c0e132c66e3b — maxpool_norm

## Status: AT_FLOOR (ratio 0.954x)

## Summary
The compiled output matches or beats the oracle performance. The oracle performs
maxpool + normalization on shape [32, 768, 7, 7] float16 tensors with two outputs
(float16 tensor and int8 indices).

## Benchmark Results
- Oracle: 18.08 us
- Compiled: 17.25 us
- Ratio: 0.954x

## Conclusion
No optimization needed. Inductor already generates code that matches or exceeds the
oracle's performance.
