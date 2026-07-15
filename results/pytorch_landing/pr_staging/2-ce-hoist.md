# PR 2 — Cross-entropy loop-invariant hoist

**Commits:** `ab29345d82d` (+ companion `9b4edfffc15`, target_logit hoist).
**Status:** DEPENDS ON mega-commit scaffolding — NOT standalone.

## Dependency (important)
Conflicts on bare base: it registers a lowering for
`inductor_prims.cross_entropy_loss_online` and uses `OnlineSoftmaxReduction`, both
introduced by the mega-commit `97385fb3273`. Cannot cherry-pick onto mainline
alone — must stack on a prerequisite PR that lands the CE-online prim + reduction
infrastructure, OR be squashed with that scaffolding into one PR.

## Summary
When the reduction dim is large, pre-computes the target-logit gather as a separate
Pointwise before the online-softmax reduction, so the target logit is a simple
buffer load inside the reduction kernel and existing loop-invariant hoisting moves
it out of the inner loop.

## Measured impact
- **Kernel-geomean +1.22pp / per-model e2e +0.54pp** (above the ±0.82pp model floor).

## Test plan
CrossEntropy forward/backward genai micros; CE-bearing training models
(Bert/Electra/Deberta train). Numerics-gated.

written with claude code
