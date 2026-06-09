# sum_sum_sum_b0debbb55e8e

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 73.5 us
- Oracle: 110.1 us
- Ratio: 0.668x (oracle 50% slower than compile)
- Oracle correctness: FAIL (output 3: shape=[512] max_diff=1.09e-01)

## Diagnosis

The oracle (cooperative split-K) is both incorrect (fails correctness) and slower than torch.compile output. The oracle's split-K approach for shape [512, 25088] produces incorrect results for one of the output tensors and runs 50% slower due to atomic contention overhead.

## Status: closed_no_gap (BAD_ORACLE + incorrect)
