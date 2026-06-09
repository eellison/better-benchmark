# sum_sum_sum_e1f39649886f

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 231.2 us
- Oracle: 911.4 us
- Ratio: 0.254x (oracle ~4x slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is catastrophically slower than torch.compile output. For the large shape [128, 401408] column reduction, the oracle's cooperative split-K approach incurs extreme atomic contention, running nearly 4x slower than Inductor's standard reduction. This is the same shape as sum_sum_sum_be777b3382b6 (which also showed BAD_ORACLE at 0.784x).

## Status: closed_no_gap (BAD_ORACLE)
