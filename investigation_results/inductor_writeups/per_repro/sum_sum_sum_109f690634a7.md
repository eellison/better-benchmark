# sum_sum_sum_109f690634a7

## Classification: BAD_ORACLE

## Oracle: oracle_resnest_dual_bn_backward.py

## Measurements

- Compiled: 177.98 us
- Oracle: 203.58 us
- Ratio: 0.874x (oracle 13% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (resnest_dual_bn_backward) is slower than torch.compile output. For the ResNeSt dual batch-norm backward with shapes [32, 256, 56, 56] and [256] channel reductions, Inductor's standard multi-output reduction with autotuned configs outperforms the oracle's manually fused kernel. The oracle's fusion strategy (combining both BN backward paths into one kernel) incurs register pressure that reduces occupancy.

## Status: closed_no_gap (BAD_ORACLE)
