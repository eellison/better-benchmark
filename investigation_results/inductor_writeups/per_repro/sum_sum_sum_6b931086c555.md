# sum_sum_sum_6b931086c555

## Classification: BAD_ORACLE

## Oracle: oracle_swin_ln_backward.py

## Measurements

- Compiled: 77.6 us
- Oracle: 110.37 us
- Ratio: 0.703x (oracle 30% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (swin_ln_backward) is significantly slower than torch.compile output. For the Swin Transformer layernorm backward with outputs [256] f32 vectors and [256, 100352] f32 matrix, Inductor's autotuned persistent reduction with CDT decisively outperforms the oracle. The oracle's cooperative fusion approach incurs excessive synchronization overhead for this relatively small reduction dimension (256 rows).

## Status: closed_no_gap (BAD_ORACLE)
