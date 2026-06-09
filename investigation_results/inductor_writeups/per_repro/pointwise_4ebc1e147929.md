# pointwise_4ebc1e147929 — Add Select

## Summary
- **Repro**: (add + select pattern)
- **Oracle**: oracle_add_select.py
- **Ratio**: 0.963x (oracle 6.05us vs compile 5.82us)
- **Status**: AT_FLOOR (compile matches or beats oracle)

## Benchmark Result
The ratio of 0.963x indicates the compiled code is already as fast as (or slightly faster
than) the oracle. No performance gap to investigate.
