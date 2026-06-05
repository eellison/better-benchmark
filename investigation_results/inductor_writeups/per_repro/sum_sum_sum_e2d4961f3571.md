# sum_sum_sum_e2d4961f3571

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 57.6 us
- Oracle: 117.7 us
- Ratio: 0.489x (oracle ~2x slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is dramatically slower than torch.compile output. For the shape [512, 32768] column reduction, the oracle's cooperative approach has excessive atomic overhead on this GPU. Inductor's standard reduction wins by more than 2x.

## Status: closed_no_gap (BAD_ORACLE)
