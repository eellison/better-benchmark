# sum_sum_sum_d32f70eaa94d

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 81.6 us
- Oracle: 268.0 us
- Ratio: 0.305x (oracle ~3x slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is dramatically slower than torch.compile output. For the [768] column reduction output, the oracle's cooperative approach with split-K partials is catastrophically inefficient on this hardware, running 3x slower than Inductor's standard reduction.

## Status: closed_no_gap (BAD_ORACLE)
