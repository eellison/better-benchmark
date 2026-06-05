# sum_sum_sum_b42d335bef54

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 26.1 us
- Oracle: 31.7 us
- Ratio: 0.824x (oracle 21% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is slower than torch.compile output. For the shape [768, 8192] column reduction, the oracle's cooperative approach adds atomic overhead that exceeds any parallelism benefit on this hardware. Inductor's standard reduction wins decisively.

## Status: closed_no_gap (BAD_ORACLE)
