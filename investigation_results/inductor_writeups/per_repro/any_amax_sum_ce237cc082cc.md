# any_amax_sum_ce237cc082cc

## Classification: CONSTANT_FOLDING

## Pattern

DistilBERT attention softmax with tautological mask: [3072, 128, 128] scores
viewed as [256, 12, 128, 128], generated arange(128) >= 0 mask (always True),
where selects 0 bias (identity add), any(eq(-inf)) guard (always False),
stable softmax, view back to [3072, 128, 128].

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 64.45 | 1.000 |
| torch.compile (cd=True, combo) | 69.31 | 1.075 |

Correctness: PASS (shape=[3072, 128, 128] f32, max_diff=5.96e-08)

## Kernel Count

- Oracle: 1 kernel (pure row-softmax, tautological mask eliminated)
- Inductor: 1 kernel (fused but with redundant mask computation)

## Root Cause

Same CONSTANT_FOLDING family: arange(128) >= 0 is always True. Inductor
does not eliminate the dead mask construction path. The gap is 1.075x
(marginal, larger shape makes it more measurable than smaller variants).

## Config Exploration

Single persistent kernel - no config improvement.

## Status: MINOR_GAP (1.075x) - constant folding of iota predicates
Same root cause as the any_amax_sum CONSTANT_FOLDING family.
