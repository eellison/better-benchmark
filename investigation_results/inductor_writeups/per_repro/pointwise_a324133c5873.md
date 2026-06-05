# pointwise_a324133c5873 (MobileViT Layout Materialize)

## Benchmark Result
- Oracle: 8.13 us
- Compiled: 8.03 us
- Ratio: 0.988x
- Status: AT_FLOOR

## Summary
Compiled code matches or beats the oracle. No regression to investigate. Inductor handles the MobileViT layout materialization pattern efficiently.
