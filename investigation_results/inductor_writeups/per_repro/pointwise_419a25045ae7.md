# pointwise_419a25045ae7 — BAD_ORACLE (ratio 0.688x)

## Summary
Oracle: `oracle_longformer_padded_layout.py` — Longformer padding + overlapping as_strided clone
Benchmark: oracle=45.89us, compile=31.58us, ratio=0.688x

## Classification: BAD_ORACLE

## Root Cause
The oracle fuses bias add, Longformer constant-pad, overlapping as_strided clone, and
final layout view into one Triton materialization kernel. However, for this shape
(384, 64, 768), Inductor's existing code generation significantly outperforms the oracle
(compile is 31% faster than oracle).

This suggests the oracle's approach (single kernel for pad + overlap + view) is
suboptimal at this size, likely due to poor occupancy or memory access patterns in
the oracle's tiling strategy compared to Inductor's optimized flat pointwise schedule.

## Kernel Count
- Oracle: 1 kernel (fused)
- Inductor: likely 1-2 kernels, but faster overall

## Config Exploration
N/A — oracle is slower; no Inductor improvement needed.

## Conclusion
Oracle is defective for this shape. Inductor already exceeds oracle performance.
No action needed.
