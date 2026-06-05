# pointwise_aa194b04264a

## Compile: 78.69us, Oracle: 84.83us, Gap: 0.928x (BAD_ORACLE)

## Classification: NO_GAP

## Root Cause

The compiled Inductor output outperforms this oracle. The oracle (BN + SiLU + pad) is 1.08x slower than compile. No investigation needed.

## Status: closed_no_gap

## Details
- Model: timm (BN + SiLU + pad)
- Shape: [128, 672, 17, 17] f32 (with NaN padding)
- Oracle attempts fused BN+SiLU+pad but Inductor's approach is faster
- Inductor already wins on this workload
