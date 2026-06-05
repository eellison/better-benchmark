# pointwise_77f01240c31d

## Classification: BAD_ORACLE

## Oracle: oracle_fused_bn_relu_resize_cat.py

## Measurements

- Compiled: 50.8 us
- Oracle: 57.0 us
- Ratio: 0.891x (oracle 12% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (fused BN + ReLU + resize + cat) is slower than torch.compile output. Inductor's autotuned pointwise codegen with shape [1, 256, 256, 256] already outperforms the hand-fused oracle kernel. No performance gap to investigate.

## Status: closed_no_gap (BAD_ORACLE)
