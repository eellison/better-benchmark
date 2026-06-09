# sum_sum_sum_cb22a50c0d9f

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 71.5 us
- Oracle: 275.4 us
- Ratio: 0.260x (oracle ~4x slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is dramatically slower than torch.compile output. For the shape [768, 16384] column reduction, the oracle's cooperative approach is catastrophically slow due to atomic contention. Inductor's standard reduction outperforms it by nearly 4x.

## Status: closed_no_gap (BAD_ORACLE)
