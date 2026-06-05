# sum_dc057cf6da21 — BAD_ORACLE

## Summary
- **Pattern**: sum
- **Ratio**: 0.872x (oracle is slower than compiled)
- **Status**: BAD_ORACLE — oracle does not represent a valid performance floor

## Benchmark
- Oracle: 6.98 us
- Compiled: 6.08 us
- Ratio: 0.872x

## Conclusion
The oracle is slower than torch.compile output. No performance gap exists; this oracle is diagnosis-only.
