# sum_fe2660cedc0a

## Compile: 44.58us, Oracle: 61.28us, Gap: 0.727x

## Classification: BAD_ORACLE

## Root Cause

The oracle (fused_layout_sum) is slower than torch.compile. The compiled code at 44.58us outperforms the oracle at 61.28us, giving a ratio of 0.727x. The oracle attempts to compute a masked f32 pointwise producer once, write the transposed-view backing storage, and accumulate the sibling f32[768] column sum through a row-major 64-row partial reduction in a single kernel. Despite avoiding the reread, Inductor's separate-kernel schedule (materialization then reduction) achieves significantly better throughput via autotuned tile configurations.

## Kernel Count
- Oracle: N/A (slower than compile)
- Inductor: baseline is already faster

## Config Exploration
No config exploration needed -- oracle is slower than compile.

## Fix Assessment: No action needed

The oracle is a BAD_ORACLE. Inductor already wins on this repro.

## Details
- Model: masked pointwise + transposed layout store + column sum
- Shape: [768, 16384] f32 pointwise output, [768] f32 column sum
- The oracle's multi-output fusion (pointwise + layout-changing store + partial reduction) has worse occupancy than Inductor's separate autotuned kernels, which compensate for the extra memory traffic with superior per-kernel throughput.
