# sum_sum_sum_bab40cbb0446

## Classification: SCHEDULER_FUSION

## Oracle: oracle_fused_layout_multi_sum.py

## Measurements

- Compiled: 169.76 us
- Oracle: 177.79 us
- Ratio: 0.955x (oracle 5% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (fused_layout_multi_sum) computes the full two-output ALBERT attention graph by materializing the scaled layout clone while also producing 128-row partial column sums, fusing eleven sibling matrix column reductions into one multi-input partial-reduction kernel, and finalizing all twelve reductions with the captured left-associated add order. Inductor emits one layout clone kernel, twelve separate partial reduction kernels, and a final add kernel.

However, on this hardware Inductor's generated code already matches or beats the oracle's fused approach. The overhead of multi-accumulator fusion in the oracle outweighs the memory traffic savings at this particular shape (4096x4096 with 12 reductions from ALBERT).

## Config Exploration

Not explored (no gap to close -- oracle is slower than compile).

## Status: closed_no_gap (AT_FLOOR)
