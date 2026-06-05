# sum_sum_sum_3e8dba104ec0

## Compile: 157.63us, Oracle: 215.07us, Gap: 0.733x (BAD_ORACLE)

## Classification: NO_GAP

## Root Cause

The compiled Inductor output significantly outperforms this oracle. The oracle (T5 embedding scatter reduce) is 1.36x slower than compile.

## Status: closed_no_gap

## Details
- Model: T5 embedding scatter reduce
- Shape: [768] f32 reductions + [32128, 768] f32 scatter output
- Oracle's fused scatter-reduce kernel cannot beat Inductor's decomposed approach
- Large vocabulary (32128) scatter likely causes high atomic contention in the oracle
- Inductor already wins on this workload; no fix needed
