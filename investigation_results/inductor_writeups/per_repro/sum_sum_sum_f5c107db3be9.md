# sum_sum_sum_f5c107db3be9

## Classification: BAD_ORACLE

## Oracle: oracle_fused_multi_sum.py

## Measurements

- Compiled: 170.88 us
- Oracle: 196.48 us
- Ratio: 0.870x (oracle 13% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (fused_multi_sum) computes the complete Repro.forward scope by materializing the required contiguous bmm layout clone, returning its transposed view, and reducing all eleven input matrices plus the cloned bmm matrix through one shared 128-row split reduction and one shared final add kernel. Inductor currently emits the layout clone, twelve separate first-stage column-sum kernels, and a final add/reduction kernel.

Despite the oracle's structural advantage (fewer kernel launches, shared memory passes), Inductor's generated code is 13% faster on this hardware. The overhead of multi-accumulator tracking in the oracle's fused kernel outweighs the benefit of reduced memory traffic at this particular shape (4096x4096 with 12 sibling sum reductions from ALBERT).

This is the same ALBERT model pattern as sum_sum_sum_bab40cbb0446 (also AT_FLOOR/BAD_ORACLE), suggesting Inductor's autotuned separate reduction approach is well-optimized for ALBERT's specific shapes.

## Config Exploration

Not explored (no gap to close -- oracle is slower than compile).

## Status: closed_no_gap (BAD_ORACLE)
