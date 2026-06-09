# pointwise_9478357f37a0 (Cat BN ReLU)

## Benchmark Result
- Oracle: 10.94 us
- Compiled: 10.85 us
- Ratio: 0.991x
- Status: AT_FLOOR

## Summary
Compiled code matches or beats the oracle. No regression to investigate. Inductor successfully fuses the cat + BN + ReLU pattern for this shape configuration.
