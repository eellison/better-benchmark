# sum_c181ecbfc1ec — AT_FLOOR

## Summary
- **Model**: hardswish_sum pattern
- **Pattern**: HardSwish activation + sum reduction producing `[1280]` output
- **Oracle**: `oracle_hardswish_sum.py`
- **Ratio**: 1.007x (oracle 9.12us vs compile 9.18us)
- **Classification**: AT_FLOOR

## Result

The ratio of 1.007x is below the 1.05 investigation threshold. Inductor's compiled output is within 0.7% of the oracle, which is essentially identical performance. The HardSwish + sum pattern is well-handled by Inductor's existing fusion and reduction codegen.

No investigation needed — Inductor is at performance floor.
