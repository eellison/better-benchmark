# sum_sum_sum_93b46c9731d8

## Classification: COOPERATIVE_SPLIT_K

## Oracle: oracle_cooperative_split_k.py

## Models: torchbench_timm_regnet_train_001, torchbench_timm_resnest_train_001

## Measurements

- Compiled (cd): 28.5 us
- Oracle (cooperative split-K): 35.8 us
- Oracle script compile (coordinate_descent_tuning): 28.6 us

## Diagnosis

Compile outperforms the oracle by 1.26x. The oracle's cooperative split-K approach for the
RegNet/ResNeSt dual-branch ReLU + batch-norm-backward (N=32, C=2240, reducing over N,H,W=7x7)
achieves 35.8us, while Inductor's compiled output achieves 28.5us.

Despite the oracle correctly identifying the pattern (coordinated sibling channel reductions with
dependent full-tensor epilogues across two BN branches), the oracle implementation cannot beat
Inductor's generic scheduling for this shape. The two-branch BN-backward with relatively small
spatial dimensions (7x7) means the reduction domain is small enough that Inductor's default
approach handles it efficiently.

## Status

AT_FLOOR -- compile outperforms oracle. No Inductor work needed for this shape.

## Done Criteria

Closed. Compiled output already faster than oracle.
