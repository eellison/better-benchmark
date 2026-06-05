# sum_sum_sum_5a4f87bbd879

## Compile: 125.82us, Oracle: 195.33us, Gap: 0.644x (BAD_ORACLE)

## Classification: NO_GAP

## Root Cause

The compiled Inductor output significantly outperforms this oracle. The oracle (T5 embedding scatter reduce) is 1.55x slower than compile.

## Status: closed_no_gap

## Details
- Model: T5 embedding scatter reduce (small hidden dim variant)
- Shape: [512] f32 reductions + [32128, 512] f32 scatter output
- Oracle's fused scatter-reduce with atomic accumulation has high contention on the large vocabulary (32128 tokens)
- Inductor's decomposed approach wins decisively
- No fix needed
