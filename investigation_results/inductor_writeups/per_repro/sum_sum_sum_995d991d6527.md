# sum_sum_sum_995d991d6527

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 67.4 us
- Oracle: 89.0 us
- Ratio: 0.758x (oracle 32% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is substantially slower than torch.compile output. For the output shapes [768] reduction from [32, 512, 768, 2], the cooperative split-K approach has too much atomic contention relative to Inductor's standard reduction strategy on this GPU.

## Status: closed_no_gap (BAD_ORACLE)
