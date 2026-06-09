# sum_sum_sum_7baf7f118798

## Classification: BAD_ORACLE

## Oracle: oracle_fused_layout_multi_sum.py

## Measurements

- Compiled: 171.94 us
- Oracle: 246.69 us
- Ratio: 0.697x (oracle 30% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (fused_layout_multi_sum) is significantly slower than torch.compile output. For the fused layout multi-sum pattern with outputs [4096, 4096] f32 and [4096] f32, Inductor's standard reduction strategy with coordinate_descent_tuning outperforms the oracle's manual fusion. The oracle's approach of combining layout transformation with multi-output reduction incurs excessive register pressure on this [4096, 4096] shape.

## Status: closed_no_gap (BAD_ORACLE)
