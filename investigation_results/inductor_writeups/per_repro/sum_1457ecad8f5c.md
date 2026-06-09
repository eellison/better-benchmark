# sum_1457ecad8f5c - Longformer Index Put Sum Transpose

## Summary
- **Pattern**: Longformer index_put + sum + transpose pattern
- **Shape**: outputs [768] (fp32) and [768, 8192] (fp32)
- **Ratio**: 1.022x (oracle 79.87 us, compiled 81.63 us)
- **Status**: AT_FLOOR - Inductor matches oracle performance

## Verdict
No action needed. The compiled kernel is within 2.2% of the oracle, well within measurement noise for this larger kernel (~80 us).
