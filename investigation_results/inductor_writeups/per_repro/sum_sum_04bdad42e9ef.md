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

AT_FLOOR -- compile BEATS oracle. No Inductor work needed for this shape.

## Re-measurement (2026-06-08)

- Oracle: 19.9 us
- Compiled: 15.94 us
- Ratio: 0.801x (BAD_ORACLE -- compile is 20% faster than oracle)

The decomposed split-K fix (d75864dea06) and aggressive split threshold (8586e404cc8) made the
compiled code significantly faster than the hand-written oracle. The oracle is now outdated.

## Done Criteria

Closed. Compiled output beats oracle performance by 20%.
