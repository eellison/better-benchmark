# sum_mean_6ea3cb4a159b — BAD_ORACLE

## Summary
- **Pattern**: BN pair mean (sum + mean)
- **Ratio**: 0.935x (oracle is slower than compiled)
- **Status**: BAD_ORACLE — oracle does not represent a valid performance floor

## Benchmark
- Oracle: 11.78 us
- Compiled: 11.01 us
- Ratio: 0.935x

## Conclusion
The oracle is slower than torch.compile output. No performance gap exists; this oracle is diagnosis-only.
