# any_amax_sum_4a963b47b28d

## Classification: CONSTANT_FOLDING

## Pattern

M2M100 attention softmax with tautological mask: [1024, 128, 128] scores
viewed as [64, 16, 128, 128], generated iota/add/unsqueeze/ge/expand mask
where iota >= 0 is always True, where selects 0 bias (identity add),
any(eq(-inf)) guard (always False), stable softmax, view back to [1024, 128, 128].

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 26.18 | 1.000 |
| torch.compile (cd=True, combo) | 28.74 | 1.098 |

Correctness: PASS (shape=[1024, 128, 128] f32, max_diff=5.96e-08)

## Kernel Count

- Oracle: 1 kernel (pure row-softmax, tautological mask proven and eliminated)
- Inductor: 1 kernel (fused but with redundant mask/guard computation)

## Root Cause

Same CONSTANT_FOLDING family: arange(128) >= 0 proved True, entire mask path
eliminated. Inductor does not simplify shape-derived predicates through
unsqueeze/expand/where before reduction scheduling.

## Config Exploration

Single persistent kernel - no config improvement.

## Status: MINOR_GAP (1.10x) - constant folding of iota predicates
Same root cause as the any_amax_sum CONSTANT_FOLDING family.
