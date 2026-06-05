# amax_sum_sum_2a28fce37306

## Status: NO GAP (DIAGNOSIS_ONLY / oracle slower)

## Measurement
- Oracle: 6.88 us
- Compiled: 6.08 us
- Combo compiled: 6.144 us
- Ratio: 0.884x (oracle is slower than compiled)
- Classification: NEW_PATTERN

## Pattern
Two-head ignore-index cross-entropy mean from GPT-J question answering.
Pattern hash `2a28fce37306`, from `hf_GPTJForQuestionAnswering_train_000`.
Shape: `[128, 2]` (128 rows, 2 classes -- very small reduction dim).

## What the Oracle Does
Custom Triton kernel computing:
- Full two-head `[128, 2]` ignore-index cross-entropy
- Online logsumexp scalar accumulators for both heads
- Gather, valid-count division, and final average in one scalar kernel

## Root Cause of BAD_ORACLE
The workload is extremely small (128 rows, 2 columns). At this scale, kernel launch
overhead dominates. Inductor's generic codegen handles the tiny reduction efficiently --
the overhead of the oracle's fused kernel (with its more complex control flow for
ignore-index handling) is not offset by any memory savings since the data easily fits
in L2 cache. The total compute is ~6 microseconds regardless of approach.

## Classification
NEW_PATTERN (per oracle docstring), but no performance gap exists. The problem size is
too small for fusion to provide meaningful benefit. No fix needed.

## Kernel Count
Not investigated (no gap to close).
