# any_amax_sum_48ee4b2cf979

## Classification: CONSTANT_FOLDING

## Pattern

MobileBERT attention softmax with tautological mask: [1024, 128, 128] scores
viewed as [256, 4, 128, 128], arange(128) >= 0 mask (always True), where selects
0 bias (identity add), any(eq(-inf)) guard (always False), stable softmax,
view back to [1024, 128, 128].

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 26.21 | 1.000 |
| torch.compile (cd=True, combo) | 28.38 | 1.083 |

Correctness: PASS (shape=[1024, 128, 128] f32, max_diff=5.96e-08)

## Kernel Count

- Oracle: 1 kernel (pure row-softmax, mask folded away)
- Inductor: 1 kernel (fused but still computes the tautological mask/guard)

## Root Cause

Same as any_amax_sum_0815d0a11243: arange(128) >= 0 is always True. Inductor
does not fold the iota predicate, emitting unnecessary mask construction and
any() guard instructions. The gap is 1.08x (marginal).

## Config Exploration

Single persistent kernel already - no multi_kernel improvement possible.

## Status: MINOR_GAP - constant folding of iota predicates
Same root cause as the CONSTANT_FOLDING family.
