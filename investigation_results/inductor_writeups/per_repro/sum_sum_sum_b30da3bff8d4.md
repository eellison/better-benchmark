# sum_sum_sum_b30da3bff8d4

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 81.8 us
- Oracle: 87.0 us
- Ratio: 0.939x (oracle 6% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is marginally slower than torch.compile output. For the output shapes [768] reduction from [32, 512, 768, 2], Inductor's standard reduction strategy already outperforms the cooperative split-K approach on this GPU.

## Status: closed_no_gap (BAD_ORACLE)
