# pointwise_55a3cc897c83 — Index Put Accumulate

## Summary
- **Repro**: (index_put with accumulate pattern)
- **Oracle**: oracle_index_put_accumulate.py
- **Ratio**: 0.989x (oracle 92.06us vs compile 91.01us)
- **Status**: AT_FLOOR (compile matches oracle)

## Benchmark Result
The ratio of 0.989x is within noise. The compiled code effectively matches the oracle
performance. No investigation needed.
