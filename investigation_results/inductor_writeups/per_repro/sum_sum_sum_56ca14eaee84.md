# sum_sum_sum_56ca14eaee84

## Compile: 219.07us, Oracle: 282.43us, Gap: 0.776x (BAD_ORACLE)

## Classification: NO_GAP

## Root Cause

The compiled Inductor output significantly outperforms this oracle. The oracle (fused multi-output reduction) is 1.29x slower than compile.

## Status: closed_no_gap

## Details
- Model: fused multi-output reduction
- Shape: [768] f32 reductions + [1, 256, 768] f32 side output + [768] f32
- Oracle's fused approach cannot beat Inductor at this shape
- Inductor already wins; no fix needed
