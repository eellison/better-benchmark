# pointwise_77c5ae8ad1b9 — oracle_attention_mask

## Summary
Oracle: 5.95us | Compiled: 5.89us | Ratio: 0.989x | Status: AT_FLOOR

## Classification: NO_GAP

The compiled code is faster than the oracle (ratio < 1.0). No investigation needed.

## Root cause
N/A - Inductor already outperforms the oracle for this [1, 16, 128, 128] attention mask pattern with expand stride=(16384, 0, 128, 1). The tensor is small and at the hardware floor.

## Kernel count
- Inductor: 1 kernel
- Oracle: 1 kernel

## Conclusion
No action needed. Inductor matches or exceeds oracle performance.
