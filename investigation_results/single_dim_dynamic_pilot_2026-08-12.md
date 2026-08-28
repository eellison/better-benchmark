# Single-dim dynamism pilot: batch vs reduction dim, warm-order sensitivity

**Date:** 2026-08-12 · **Hardware:** B200 · **Branch:** dynamic-shapes-capture

Maintainer framing: it is rare for ALL dims to be dynamic — usually exactly
one is (dynamic batch is one common case, dynamic seq-len another). This
pilot captures blocks with exactly ONE dynamic dim
(`scripts/pilot_single_dim_dynamic.py`, scratch-only) and answers:

1. dynamic vs static at the same eval points, when only one dim is dynamic;
2. warm-order/compile-history sensitivity ("run first at small, then time at
   8 — does that mess up perf?") via the new repro-bench `--prewarm`
   (the FIRST warm binding is the hint inductor tunes the general kernel at);
3. the batch=1-then-8 story, mechanically.

Sweeps: repro self-bench (`--bind`, `--dynamic --prewarm ... --bind`), GPU
lock, CUDAGraph, do_bench min. Raw: `single_dim_dynamic_pilot_raw_2026-08-12.json`.

## Results (dyn/static ratio per eval point; hint = capture shape)

| family (one dynamic dim) | dynamic dim | s=2 | 8 (hint) | 64 | 512 | kernels |
|---|---|---:|---:|---:|---:|---|
| FFN LayerNorm (feature 1024 static) | batch | 0.99 | 0.99 | 0.99 | **0.99** | per→per |
| FFN GELU chain | batch | 0.98 | 0.99 | 1.17 | 1.53 | poi→poi |
| GroupNorm+ReLU (spatial 16×16 static) | batch | 0.98 | 0.82 | 0.88 | 1.26 | per→per |

| family | dynamic dim | K=64 | 256 (hint) | 1024 | 4096 | kernels |
|---|---|---:|---:|---:|---:|---|
| attention softmax [16,8,128,K] | K = **reduction** dim | 2.08 | 1.56 | 1.63 | 1.44 | per→red |

(Absolute µs in the raw JSON. The GroupNorm row exists because of the
finding-4 fix; ratios ≤1 at 4–64 mean the general kernel's config beats
some per-point static tunings — no dynamism penalty until the mild
large-batch drift both pointwise-heavy families show at 512.)

## Findings

1. **Which dim is dynamic is what matters — not how many.** LayerNorm with
   dynamic batch is FREE at every point (0.99×, and the persistent
   `per_*` kernel SURVIVES because the reduction numel — the static 1024
   feature — is what the persistent heuristic needs). The same reduction op
   with the reduction dim dynamic (softmax over K) pays 1.4–2.1× and drops
   to the looped `red_*` kernel. The earlier all-dims-dynamic sweep's 2–5×
   (dynamic_vs_static_eval_points_2026-08-12.md) is the reduction-dim
   penalty, not a general "dynamism tax": marking only batch avoids it.
   (Pointwise with dynamic batch drifts up at large batch — 1.53× at 512 —
   grid-sizing overhead worth a look, but small/mid batches are free.)

2. **Warm order / first-seen shape does NOT matter for the general kernel**
   (maintainer question 2). small-first (`--prewarm 2 then 8`) vs hint-first
   (`--prewarm 8 then 16`) agree within ±2% at every point of every family
   (one 12% outlier: LayerNorm small-first at batch=64; not reproducible
   direction, treated as run noise). Once the general kernel exists, WHICH
   shapes created it doesn't change its per-point performance in these
   families.

3. **batch=1 first is safe** (maintainer question 3, mechanism probe):
   with the batch dim marked dynamic, running 1 → 8 → 16 yields
   `unique_graphs=2` — batch=1 gets its OWN specialized graph (dynamo's 0/1
   rule), 8 and 16 share one general graph. A batch=1 warmup costs one
   extra compile but never becomes the general kernel's tuning hint and
   does not degrade later batch=8 performance.

4. **Repro-fidelity gap for symint-input families under `--dynamic` —
   found AND fixed** (GroupNorm-batch-dyn case): the captured region takes
   the model's `x.size(0)` as a SYMINT INPUT; the standalone repro lifts it
   to a plain Python int argument; dynamo specializes int args per compile,
   and the reshape's `infer_size` then PINS the marked batch dim to that
   constant (`Eq(16384*s67, 65536)` → s67=4) → `ConstraintViolationError`,
   at any binding. The ORIGINAL model never sees this — its symint is
   derived from the tensor and stays symbolic. FIX (repro_harness),
   FALLBACK-ONLY: the `--dynamic` bench first builds the faithful raw-int
   artifact; only when that raises ConstraintViolation does it rebuild
   with symint inputs re-derived from their source tensor dims INSIDE the
   traced forward (`_symint_derivations_for_repro` + `_DerivedSymintRepro`:
   each `['I',hint,expr]` slot whose expr's root symbols are readable off
   a tensor input's dims becomes `tensor.size(d)` arithmetic; the compiled
   callable takes only the remaining args), announcing the substitution
   loudly. FALLBACK-ONLY matters: deriving unconditionally CHANGES
   inductor's fusion for symint-VALUE families (var_mean fused 1 kernel
   vs the model's 2 — the same unfaithfulness as compile_fx-direct,
   caught by test_dynamic_default_measures_general_kernel_gpu when the
   first version derived always). Verified: the pin-hazard family now
   runs at every binding including the previously-fatal 2 and 4 (table
   above) with the persistent kernel surviving; value-consuming and
   symint-free families keep their exact pre-fix artifacts.

5. **Prior curve re-verified on the current harness** (the earlier sweep
   accidentally imported a stale June harness via sys.path fallthrough —
   now fixed: emitted repros carry the generating repo root as an appended
   sys.path fallback). Spot-check, var_mean 16×16 / 64×64: static 8.32 /
   32.83 µs (was 8.42 / 32.45), dynamic 21.34 / 147.10 (was 18.05 /
   167.81) — same kernel structure, same story, dynamic rows vary ±15%
   run-to-run.

## Corpus/product implication

For the eventual dynamic corpus, per-family metadata should record WHICH
dims are dynamic (parallel vs reduction) — it predicts the cost class: a
batch-dynamic family prices like its static twin; a reduction-dynamic
family needs the persistent-reduction follow-up before its floor is
meaningful.
