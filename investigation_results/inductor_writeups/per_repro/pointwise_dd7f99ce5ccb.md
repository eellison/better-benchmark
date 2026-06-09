# pointwise_dd7f99ce5ccb — Reformer Cat Index

## Status: BAD_ORACLE (ratio = 0.918x)

## Oracle Description
The oracle fuses cat/view/select/index/unsqueeze into one gather that reads arg0 for columns 0..63 and arg1 for columns 64..255, avoiding materializing the concatenation before applying the indexed row gather.

## Classification: SCHEDULER_FUSION (oracle slower than compile)

## Benchmark Results
- Oracle: 7.01 us
- Compiled: 6.43 us
- Ratio: 0.918x (compile is FASTER)

## Root Cause
Despite the oracle's claim that Inductor materializes concatenation before gather, in practice the compiled kernel is faster. This suggests Inductor already optimizes this pattern well (possibly fusing through the concat), or the oracle's branch-based load pattern (conditional on `cols < 64`) introduces warp divergence that hurts more than the extra memory traffic from materializing the concat.

## Kernel Count
Both effectively 1 kernel. Inductor's approach wins on this fp16[1,4096,256] shape.

## Config Exploration
No config changes needed — compile already beats oracle.

## Conclusion
Oracle is suboptimal for this shape. No action needed.
