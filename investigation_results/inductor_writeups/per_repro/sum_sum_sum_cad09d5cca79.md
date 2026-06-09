# sum_sum_sum_cad09d5cca79

## Classification: BAD_ORACLE

## Oracle: oracle_maxpool_backward_scatter_reduce.py

## Measurements

- Compiled: 300.9 us
- Oracle: 577.4 us
- Ratio: 0.521x (oracle ~2x slower than compile)
- Oracle correctness: FAIL (output 1: shape=[256] max_diff=4.00)

## Diagnosis

The oracle (maxpool backward scatter reduce) is both incorrect and dramatically slower than torch.compile output. The oracle produces wrong results for one output tensor and takes nearly 2x longer. The scatter-reduce fusion in the oracle is counterproductive for this workload on this hardware.

## Status: closed_no_gap (BAD_ORACLE + incorrect)
