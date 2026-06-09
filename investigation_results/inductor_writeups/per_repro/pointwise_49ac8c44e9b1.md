# pointwise_49ac8c44e9b1 — Embedding Oracle

## Summary
- **Repro**: (embedding pattern)
- **Oracle**: oracle_embedding.py
- **Ratio**: 0.849x (oracle 7.01us vs compile 5.95us)
- **Status**: BAD_ORACLE (compile already faster than oracle)

## Benchmark Result
The compiled code is already faster than the oracle (ratio < 1.0). No investigation needed.
Inductor's generated code for this pattern is already optimal or better than the hand-written
oracle approach.
