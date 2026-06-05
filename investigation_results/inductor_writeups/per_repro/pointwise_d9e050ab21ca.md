# pointwise_d9e050ab21ca — GPT-J RoPE Pair

## Status: AT_FLOOR (ratio = 1.004x)

## Oracle Description
The oracle computes the complete GPT-J two-branch RoPE (rotary position embedding) graph in one Triton pointwise kernel. It handles both strided input permutes, coefficient expansion, pairwise slice_scatter rotate-half on the first 64 dims, preservation of remaining 192 dims, and two final transposed f32[4096,128] view returns.

## Classification: SCHEDULER_FUSION (already resolved)

## Benchmark Results
- Oracle: 7.42 us
- Compiled: 7.46 us
- Ratio: 1.004x

## Root Cause
The oracle diagnosis claims Inductor cannot fuse the duplicated RoPE rotate-half plus tail-copy assembly into one direct write. However, benchmarks show Inductor's compiled output already matches the oracle performance (within noise). The compile path likely already achieves similar fusion or the workload is small enough that any kernel count difference is dominated by launch overhead floor.

## Kernel Count
Both oracle and compiled produce equivalent throughput at ~7.4 us, indicating 1 kernel each (or equivalent effective performance).

## Config Exploration
No config changes needed — already at floor.

## Conclusion
No action required. Inductor matches oracle performance on this workload.
