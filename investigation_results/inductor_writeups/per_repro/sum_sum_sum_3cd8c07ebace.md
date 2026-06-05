# sum_sum_sum_3cd8c07ebace

## Compile: 475.14us, Oracle: 603.97us, Gap: 0.787x (BAD_ORACLE)

## Classification: NO_GAP

## Root Cause

The compiled Inductor output significantly outperforms this oracle. The oracle (maxpool + BN scatter reduce) is 1.27x slower than compile.

## Status: closed_no_gap

## Details
- Model: maxpool + BN + scatter reduce
- Shape: [128, 64, 35, 35] f32 + [64] f32 reductions
- Oracle's fused kernel cannot beat Inductor's decomposed multi-kernel approach
- Inductor already wins on this workload; no fix needed
