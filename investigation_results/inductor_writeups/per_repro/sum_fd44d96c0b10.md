# sum_fd44d96c0b10 — AT_FLOOR

## Summary
- **Pattern**: structured pool upsample backward reduce
- **Ratio**: 0.960x
- **Status**: AT_FLOOR — Inductor already matches oracle performance

## Benchmark
- Oracle: 103.17 us
- Compiled: 99.01 us
- Ratio: 0.960x

## Conclusion
Inductor already generates near-optimal code for this pattern. No actionable performance gap.
