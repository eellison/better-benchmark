# pointwise_fd8663effbda - Broadcast Hardsigmoid Mul

## Summary
- **Pattern**: Broadcast hardsigmoid followed by multiply
- **Shape**: [256, 960, 7, 7] (fp16)
- **Ratio**: 0.907x (oracle 15.14 us, compiled 13.73 us)
- **Status**: BAD_ORACLE - compiled is already faster than the oracle

## Verdict
No action needed. Inductor outperforms the oracle by ~9%. The oracle's hand-written kernel is suboptimal for this pattern.
