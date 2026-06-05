# sum_sum_sum_bc73f5ef21f6

## Classification: BAD_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Compiled: 94.1 us
- Oracle: 114.7 us
- Ratio: 0.820x (oracle 22% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (cooperative split-K) is slower than torch.compile output. For the output shapes [1, 1, 768] and [768], the cooperative split-K approach is counterproductive on this hardware -- Inductor's standard persistent reduction outperforms it by 22%.

## Status: closed_no_gap (BAD_ORACLE)
