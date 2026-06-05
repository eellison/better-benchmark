# sum_sum_sum_bebd11dba544

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 116.6 us
- Oracle: 182.1 us
- Ratio: 0.640x (oracle 56% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is dramatically slower than torch.compile output. For the shape [768, 25216] column reduction, the oracle's cooperative approach is heavily penalized by atomic contention on this GPU architecture. Inductor's standard reduction kernels are far superior.

## Status: closed_no_gap (BAD_ORACLE)
