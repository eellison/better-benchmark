# sum_sum_ab550e4e948e

## Classification: SCATTER_REDUCE

## Oracle: oracle_maxpool_bn_scatter_reduce.py

## Model: torchbench_timm_vovnet_train_001

## Measurements

- Compiled (cd): 29.6 us
- Oracle (triton): 537.1 us

## Diagnosis

Compile dramatically outperforms the oracle (18.1x faster). The oracle attempts a structured
scatter-reduce for the VoVNet max-pool-backward `scatter_add` with downstream ReLU/BatchNorm-backward,
but at 537us it is catastrophically worse than Inductor's 29.6us.

The oracle likely has a bug or severe inefficiency in its scatter-reduce implementation for this
shape (C=768, spatial 14x14, batch 32). The max-pool indices create irregular scatter patterns
that the oracle's tiling strategy handles very poorly, while Inductor's generic approach with
separate scatter_add and reduction kernels achieves good performance through better occupancy.

## Status

AT_FLOOR -- compile outperforms oracle by 18.1x. BAD_ORACLE: oracle implementation is catastrophically
slower than compile; needs complete rewrite.

## Done Criteria

Closed. Compiled output already far faster than oracle.
