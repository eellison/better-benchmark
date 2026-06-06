# any_amax_sum_0815d0a11243

## Classification: CONSTANT_FOLDING

## Pattern

Blenderbot attention softmax with tautological mask: [512, 128, 128] scores
viewed as [16, 32, 128, 128], arange(128) >= 0 mask (always True), where selects
0 bias (identity add), any(eq(-inf)) guard (always False), stable softmax,
view back to [512, 128, 128].

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 16.54 | 1.000 |
| torch.compile (cd=True, combo) | 17.70 | 1.070 |

Correctness: PASS (shape=[512, 128, 128] f32, max_diff=5.96e-08)

## Kernel Count

- Oracle: 1 kernel (pure row-softmax, mask folded away)
- Inductor: 1 kernel (fused but still computes the mask/guard)

## Root Cause

The arange(128) >= 0 predicate is always True, making the mask tautological.
The oracle proves this algebraically and emits a pure softmax kernel. Inductor
does not constant-fold shape-derived iota predicates, so it still generates
mask construction + any() guard instructions within the fused kernel.

The gap (1.07x) is at measurement noise floor for this small shape (16.5 us).

## Config Exploration

Single persistent kernel already - no multi_kernel improvement possible.

## Status: MINOR_GAP - constant folding of iota predicates would close this
Same root cause as any_amax_sum_97b43144672a family.
