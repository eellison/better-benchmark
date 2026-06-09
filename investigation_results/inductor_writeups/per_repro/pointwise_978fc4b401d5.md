# pointwise_978fc4b401d5 (Longformer Layout)

## Benchmark Result
- Oracle: 7.81 us
- Compiled: 7.65 us
- Ratio: 0.980x
- Status: AT_FLOOR

## Summary
Compiled code matches or beats the oracle. No regression to investigate. Inductor handles this Longformer layout materialization pattern efficiently.
