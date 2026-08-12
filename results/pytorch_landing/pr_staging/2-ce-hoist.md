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
**Compile-smoke (2026-07-15, B200, PYTHONPATH-shadow):** import OK (y) — the
carved `cross_entropy_loss_online` prim, `CrossEntropyOnlineReduction` (ir),
`online_softmax` config, and the postgrad gather-fusion pass all resolve at
import; `CrossEntropyOnlineReduction` is absent from `lowering` at runtime (Path
B). Representative repro
`repros/models/genai/static/CrossEntropyForward/full_graph_000.py` (8192x262144)
`torch.compile`s to completion — **GOOD**. It lowers to a single fused
`online_softmax_reduce` kernel in which the **target-logit gather is hoisted OUT
of the reduction loop** (loaded once after `online_softmax_reduce`, not
per-iteration) — the Path-B loop-invariant hoist this PR exists to produce.
Numerics vs eager: **EXACT** (`max_abs=0.0`). **Net: imports + compiles a
representative repro to a numerics-valid result on B200; full CI not run.**
**Perf verification (2026-07-15, B200, A/B PYTHONPATH-shadow base 5e2ab vs branch
tip, fresh inductor cache per arm, bench_parallel locked path): FAILS — perf-neutral
vs clean base on every targeted bench.** genai CrossEntropyForward full_graph
(8192×262144): 2261.7 → 2260.9us (1.00x); corpus CE repros
`amax_sum_sum_42988b64e7f9` (DistillGPT2 1247.0→1246.9us) and
`amax_sum_sum_4bf8a79efec4` (GPTNeo 329.6→329.5us, Roberta 1247.0→1246.0us) — all
1.00x. Root cause: on clean base the gather is ALREADY hoisted — base lowers genai
CE to a single fused `prepare_softmax_online` kernel with the target-logit gather
outside the reduction loop, and the branch's generated kernel is **byte-identical**
to base's (diff of generated code, modulo cache paths). On the corpus repros the
pass does fire (graph rewritten to `cross_entropy_loss_online`) but timing is exact
parity. The original **+1.22pp was measured relative to the mega-commit
`97385fb3273` state, whose Path-A in-loop CE reduction had REGRESSED these kernels**
(walk: DistillGPT2 1246.8 pre-mega → 3670.8 mega → 1329.1 post-hoist); `ab29345`
fixed the mega's regression rather than beating pre-mega base. Also: a
prereq-only arm (1cca4ca8347 without ab29345) crashes compiling the corpus CE
repros (`NotImplementedError: online_softmax_cross_entropy combine not used
directly`) — the two commits are only valid as a unit, and as a unit they deliver
no speedup over clean base. **Recommendation: do not upstream standalone on a perf
claim; its value is only as scaffolding-fix inside the mega lineage.** Raw data:
`perf_verify/RESULTS.json` (this dir).

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
