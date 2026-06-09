# sum_2ccb3947c2c9 — AT_FLOOR

## Summary
- **Model**: structured_pool_upsample_backward_reduce
- **Pattern**: pool backward + upsample + channel reduction
- **Ratio**: 1.022x (within noise)
- **Status**: AT_FLOOR — Inductor already near-optimal

## Operation
Takes pooled gradient, applies structured pool upsample backward, and reduces to `[768]` channel sum
plus `[768, 8192]` intermediate output.

## Benchmark
- Oracle: 133.15 us
- Compiled: 136.10 us
- Ratio: 1.022x

## Conclusion
The 2.2% gap is within measurement noise. Inductor generates effective code for this pattern.
No actionable gap.
