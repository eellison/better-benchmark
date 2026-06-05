# sum_sum_sum_c13919905d9b

## Classification: BAD_ORACLE

## Oracle: oracle_structured_pool_upsample_backward_reduce.py

## Measurements

- Compiled: 62.4 us
- Oracle: 82.9 us
- Ratio: 0.753x (oracle 33% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (structured pool upsample backward reduce) is substantially slower than torch.compile output. For the shape [128, 1408, 7, 7] with channel reduction, Inductor's autotuned reduction kernels outperform the oracle's structured approach. The oracle's pool/upsample backward fusion strategy is counterproductive on this GPU.

## Status: closed_no_gap (BAD_ORACLE)
