# sum_sum_sum_985bf52428b3

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 107.6 us
- Oracle: 179.2 us
- Ratio: 0.600x (oracle 67% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is substantially slower than torch.compile output on this hardware. The oracle's cooperative reduction strategy for shape [1, 256, 768] with 768 column reduction is counterproductive -- the reduction domain is too small for split-K to benefit, and the atomic overhead dominates.

## Status: closed_no_gap (BAD_ORACLE)
