# sum_sum_sum_3ec568a7ba04

## Classification: COOPERATIVE_SPLIT_K

## Oracle: oracle_cooperative_split_k.py

## Models: timm_dm_nfnet_f0_train (3 instances), torchbench_timm_nfnet_train_001

## Measurements

- Compiled (cd): 33.7 us
- Oracle (cooperative split-K): 27.6 us
- Oracle script compile (coordinate_descent_tuning): 33.8 us

## Diagnosis

The oracle outperforms compile by 1.22x. This is the one case among these 10 where the oracle
genuinely beats the compiled output. The oracle cooperatively split-K-reduces the NFNet
sigmoid-gated spatial product for both a scalar all-channel sum and a [1536] sigmoid-gradient
channel vector (N=128, C=1536, reducing over N,H,W).

The gap exists because Inductor schedules the scalar sum, spatial reduction, sigmoid-derivative
chain, and final channel sum as separate generic kernels over materialized intermediates. The
cooperative split-K approach coordinates these reductions from the same producer tiles, avoiding
redundant reads.

The fix requires teaching Inductor to form a cooperative split-K multi-output reduction when
one sibling output is a scalar all-channel sum and another is a per-channel reduction with a
dependent sigmoid-gradient epilogue.

## Status

COOPERATIVE_SPLIT_K -- genuine 1.22x gap. Needs Inductor cooperative split-K template.

## Done Criteria

- Inductor implements cooperative split-K multi-output reduction template that handles mixed-rank
  outputs (scalar + per-channel) from the same reduction domain.
- Compile achieves oracle parity (<=28us) for this shape.
