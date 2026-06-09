# sum_mean_be3cca3d51c6 — AT_FLOOR

## Summary
- **Pattern**: ConvNeXtV2 GRN channels-last (sum + mean)
- **Ratio**: 0.999x
- **Status**: AT_FLOOR — Inductor already matches oracle performance

## Benchmark
- Oracle: 62.34 us
- Compiled: 62.27 us
- Ratio: 0.999x

## Conclusion
Inductor already generates near-optimal code for this pattern. No actionable performance gap.
