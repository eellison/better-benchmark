# sum_sum_sum_bfac9a5afa42

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 32.5 us
- Oracle: 34.7 us
- Ratio: 0.935x (oracle 7% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is marginally slower than torch.compile output. For the shape [1024, 6272] column reduction, Inductor's standard reduction already matches or beats the cooperative approach. No gap to investigate.

## Status: closed_no_gap (BAD_ORACLE/AT_FLOOR)
