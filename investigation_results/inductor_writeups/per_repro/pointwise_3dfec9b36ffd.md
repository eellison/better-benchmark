# pointwise_3dfec9b36ffd

## Compile: 6.91us, Oracle: 8.0us, Gap: 0.864x (BAD_ORACLE)

## Classification: NO_GAP

## Root Cause

The compiled Inductor output outperforms this oracle. The oracle (segment mask multi output) is 1.16x slower than compile. No investigation needed.

## Status: closed_no_gap

## Details
- Model: segment mask multi output
- Shape: [4, 1, 512, 512] bf16 (8 outputs)
- Oracle produces slower results than Inductor's decomposed approach
- Inductor already wins on this workload
