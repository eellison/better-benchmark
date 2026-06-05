# pointwise_bd6cd7133361 — relu_dropout

## Status: BAD_ORACLE (ratio 0.867x)

## Summary
The compiled output is FASTER than the oracle. The oracle performs relu + dropout
with stochastic output (output 0 is stochastic, output 1 is exact bool mask).
The oracle is not a true floor for this repro.

## Benchmark Results
- Oracle: 16.38 us
- Compiled: 14.21 us
- Ratio: 0.867x (compiler wins)

## Conclusion
No optimization needed. The oracle is suboptimal — the compiler already produces
better code. This oracle should be updated or marked as non-authoritative.
