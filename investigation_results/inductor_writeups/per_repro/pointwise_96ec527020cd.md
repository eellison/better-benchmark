# pointwise_96ec527020cd (RoPE QK Expand)

## Benchmark Result
- Oracle: 26.21 us
- Compiled: 26.11 us
- Ratio: 0.996x
- Status: AT_FLOOR

## Summary
Compiled code matches or beats the oracle. No regression to investigate. Inductor already generates efficient code for the RoPE (Rotary Position Embedding) QK expansion pattern at this shape.
