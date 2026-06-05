# pointwise_9edaf9e5b4c4 (OPT Scale Permute)

## Benchmark Result
- Oracle: 7.07 us
- Compiled: 7.04 us
- Ratio: 0.995x
- Status: AT_FLOOR

## Summary
Compiled code matches or beats the oracle. No regression to investigate. Inductor handles the OPT model's scale + permute pattern efficiently.
