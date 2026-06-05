# sum_sum_sum_be777b3382b6

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 231.0 us
- Oracle: 294.6 us
- Ratio: 0.784x (oracle 28% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is substantially slower than torch.compile output. For the large shape [128, 401408] column reduction, the oracle's split-K approach incurs excessive atomic contention. Inductor's standard reduction kernels are well-suited for this large reduction domain.

## Status: closed_no_gap (BAD_ORACLE)
