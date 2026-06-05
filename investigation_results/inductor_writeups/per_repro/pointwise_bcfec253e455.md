# pointwise_bcfec253e455 — ema_copy

## Status: AT_FLOOR (ratio 1.018x)

## Summary
The compiled output matches the oracle performance. The oracle performs an EMA
(exponential moving average) copy operation on shape [128] float32 tensors.

## Benchmark Results
- Oracle: 5.25 us
- Compiled: 5.34 us
- Ratio: 1.018x

## Conclusion
No optimization needed. Inductor already generates code that matches the oracle's
performance within noise margin.
