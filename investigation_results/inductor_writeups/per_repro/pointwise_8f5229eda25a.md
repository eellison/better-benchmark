# pointwise_8f5229eda25a (Attention Mask Layout)

## Benchmark Result
- Oracle: 5.7 us
- Compiled: 5.5 us
- Ratio: 0.966x
- Status: AT_FLOOR

## Summary
Compiled code matches or beats the oracle. No regression to investigate. The oracle validates layout correctness (stride matching) for an attention mask expansion pattern. Inductor already generates optimal code for this pattern.
