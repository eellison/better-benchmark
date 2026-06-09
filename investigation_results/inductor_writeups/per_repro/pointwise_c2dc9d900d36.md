# pointwise_c2dc9d900d36 — dropout_add_view

## Status: AT_FLOOR (ratio 0.960x)

## Summary
The compiled output matches or beats the oracle performance. The oracle performs
dropout + add + view with stochastic output.

## Benchmark Results
- Oracle: 11.23 us
- Compiled: 10.78 us
- Ratio: 0.960x

## Conclusion
No optimization needed. Inductor already generates code that matches or exceeds the
oracle's performance.
