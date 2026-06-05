# any_mean_fde44ec669a9

## Summary
- **Model**: T5 RMSNorm aliases pattern (inference)
- **Pattern**: RMSNorm with aliased outputs (3 identical fp16 [2048,512] tensors)
- **Ratio**: 1.037x (oracle 11.17us vs compile 11.58us)
- **Classification**: AT_FLOOR
- **Status**: No actionable gap

## Benchmark Results
- Oracle: 11.17 us
- Compiled: 11.58 us
- Ratio: 1.037x

## Root Cause

The ratio is below the 1.05x threshold. Inductor is essentially at the performance floor for this workload. The oracle uses a T5 RMSNorm fusion with aliased outputs, and Inductor already generates near-optimal code for this pattern.

## Kernel Count
- Inductor: 1 kernel (fused)
- Oracle: 1 kernel

## Config Exploration
No further config exploration needed - already at floor.

## Conclusion
No investigation needed. Inductor matches the oracle performance within noise margin.
