# mean_1b78c675fe8a

## Summary
- **Model**: Seeded dropout + residual + RMSNorm (inference)
- **Pattern**: Seeded dropout -> residual add -> RMSNorm -> f32 [4096,512] output
- **Ratio**: 0.995x (oracle 13.6us vs compile 13.54us)
- **Classification**: AT_FLOOR
- **Status**: No actionable gap (stochastic ops present)

## Benchmark Results
- Oracle: 13.6 us
- Compiled: 13.54 us
- Ratio: 0.995x

## Notes
This repro contains stochastic ops (seeded dropout). The oracle cannot establish exact equality due to the stochastic nature of the computation. The compile is actually marginally faster than the oracle (ratio < 1.0), indicating Inductor is already at or beyond optimal for this pattern.

## Conclusion
No investigation needed. Inductor matches or beats the oracle performance.
