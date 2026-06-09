# pointwise_d76b39ac2742 - Embedding

## Summary
- **Status**: BAD_ORACLE (ratio 0.933x -- Inductor is faster than oracle)
- **Category**: No regression
- **Oracle**: oracle_embedding.py

## Benchmark Results
- Oracle: 6.21 us
- Compiled: 5.79 us
- Ratio: 0.933x

## Root Cause Analysis
The oracle is slower than Inductor's compiled output for this embedding lookup operation. Inductor generates efficient gather code that outperforms the hand-written oracle.

## Conclusion
BAD_ORACLE -- Inductor already wins by ~7%. No action required.
