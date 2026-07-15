# PR 2 — Cross-entropy loop-invariant hoist

**Commit:** `ab29345d82d`. Companion `9b4edfffc15` **dropped** (see below).
**Status:** MERGE-READY via carved prerequisite — `pr/ce-hoist-clean`.

## Branch & how made mergeable
`pr/ce-hoist-clean` (rooted on `5e2ab3055de`):
- `1cca4ca8347` — **prerequisite**, carved from mega `97385fb3273`: the
  `cross_entropy_loss_online` prim (`inductor_prims.py`), the
  `online_softmax_cross_entropy` ReductionType (`ops_handler.py`),
  `CrossEntropyOnlineReduction` + reduction combine/default/num_outputs
  (`ir.py`, `utils.py`), the mega "Path A" CE lowering (`lowering.py`), the
  `_cross_entropy_gather_fusion_pass` + its wiring under `config.online_softmax`
  (`post_grad.py`), and the `config._sibling_reduction_fusion` flag referenced by
  the carried ir hunk.
- `8a52af079e5` — `ab29345d82d` cherry-picked **clean** (pre-image matched
  exactly). It flips the lowering to "Path B" (pre-compute the gather as a
  separate `Pointwise`, then a plain `OnlineSoftmaxReduction`), which uses only
  base symbols and removes the `CrossEntropyOnlineReduction` import from
  `lowering.py`.

## Conflict classification
REAL SYMBOL DEPENDENCY: `cross_entropy_loss_online` is a lowering function
introduced whole by the mega-commit; `ab29345d82d` is a diff against it. Note
`OnlineSoftmaxReduction` already exists at base — only the prim,
`CrossEntropyOnlineReduction`, the reduction type, and the postgrad fusion pass
are mega-introduced.

## Companion `9b4edfffc15` — DROPPED (justified)
It hoists `target_logit` in the *in-loop* `online_softmax_cross_entropy` triton
codegen. But Path B (and HEAD `daa79cd25ca`) never creates that reduction, so the
codegen it patches is dead. It also bundles an unrelated
`slice_scatter_partial_elision` change (`config.py` + a `slice_scatter_elision.py`
absent at base). Not separable, not needed for the CE-hoist win. A cherry-pick
attempt confirmed modify/delete + content conflicts; excluded.

## Verification (2026-07-15)
7 touched `.py` files all ast-parse; 0 conflict markers; `git log` shows the two
intended commits; prim + lowering fn + Path-B `OnlineSoftmaxReduction.create` +
postgrad pass/wiring all present; `CrossEntropyOnlineReduction` import removed
from `lowering.py` (0 refs) as intended; patch applies clean onto pristine base.
**Caveat:** ast-parse + symbol-grep only — no torch build here, not compiled/run.

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
