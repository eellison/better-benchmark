# sum_sum_sum_e06c904e550e

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 50.1 us
- Oracle: 64.3 us
- Ratio: 0.779x (oracle 28% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is slower than torch.compile output. For [512] column reductions, Inductor's standard reduction strategy outperforms the cooperative approach. The atomic overhead of split-K exceeds any parallelism benefit for this shape.

## Status: closed_no_gap (BAD_ORACLE)
