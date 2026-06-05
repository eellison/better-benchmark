# pointwise_75503c474ccc — oracle_index_put

## Summary
Oracle: 30.37us | Compiled: 29.54us | Ratio: 0.973x | Status: AT_FLOOR

## Classification: NO_GAP

The compiled code is faster than the oracle (ratio < 1.0). No investigation needed.

## Root cause
N/A - Inductor already outperforms the oracle for this [50265, 768] index_put pattern. The operation is memory-bandwidth bound and Inductor's generated code is already optimal.

## Kernel count
- Inductor: 1 kernel
- Oracle: 1 kernel

## Conclusion
No action needed. Inductor matches or exceeds oracle performance.
