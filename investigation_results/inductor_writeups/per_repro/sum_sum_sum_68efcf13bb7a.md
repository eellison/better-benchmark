# sum_sum_sum_68efcf13bb7a

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 113.6 us
- Oracle: 524.2 us
- Ratio: 0.217x (oracle 4.6x slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is dramatically slower than torch.compile output on this hardware. The oracle's cooperative reduction strategy is counterproductive for this shape [768, 16384], likely due to excessive atomic contention or suboptimal split factor for this GPU's SM count.

## Status: closed_no_gap (BAD_ORACLE)
