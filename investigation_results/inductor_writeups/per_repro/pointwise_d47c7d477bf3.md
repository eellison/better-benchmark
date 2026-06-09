# pointwise_d47c7d477bf3 - Hardswish

## Summary
- **Status**: AT_FLOOR (ratio 1.011x)
- **Category**: No actionable gap
- **Oracle**: oracle_hardswish.py

## Benchmark Results
- Oracle: 5.82 us
- Compiled: 5.89 us
- Ratio: 1.011x

## Root Cause Analysis
Inductor is effectively at parity with the oracle for this hardswish activation pattern. The difference is only 0.07 us -- well within measurement noise. Inductor's lowering of hardswish (x * clamp(x+3, 0, 6) / 6) is already optimal.

## Conclusion
AT_FLOOR -- no actionable performance gap. Inductor matches the oracle for hardswish.
