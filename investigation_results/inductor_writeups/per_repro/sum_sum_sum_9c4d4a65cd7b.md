# sum_sum_sum_9c4d4a65cd7b

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 94.0 us
- Oracle: 114.5 us
- Ratio: 0.821x (oracle 22% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is slower than torch.compile output. For the output shapes [1, 1, 768] and [768] reductions, the cooperative split-K approach cannot beat Inductor's standard persistent reduction on this hardware. The split-K atomic overhead exceeds any parallelism benefit.

## Status: closed_no_gap (BAD_ORACLE)
