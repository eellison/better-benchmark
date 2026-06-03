# sum_sum_04bdad42e9ef

## Classification: COOPERATIVE_SPLIT_K

## Oracle: oracle_cooperative_split_k.py

## Model: torchbench_timm_efficientnet_train_001

## Measurements

- Compiled (cd): 21.1 us
- Oracle (cooperative split-K): 22.2 us
- Oracle script compile (combo_looped_cd): 20.3 us

## Diagnosis

Compile is at parity with the oracle. The oracle's cooperative split-K approach for the EfficientNet
SiLU-gradient + batch-norm-backward achieves 22.2us, while Inductor's compiled output achieves
21.1us (cd) and 20.3us (combo_looped_cd from the oracle's own benchmark).

The oracle diagnoses the gap as needing a cooperative split-K multi-output reduction template that
coordinates compatible sibling channel reductions (N=32, C=1280, reducing over N,H,W). However,
Inductor already matches or beats the oracle for this shape, meaning the current scheduler's
generic reduction strategy is sufficient.

## Status

AT_FLOOR -- compile matches oracle. No Inductor work needed for this shape.

## Done Criteria

Closed. Compiled output already at oracle performance.
