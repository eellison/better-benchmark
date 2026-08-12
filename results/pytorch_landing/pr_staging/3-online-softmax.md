# PR 3 — Fast combine for online-softmax loops

**Commit:** `a26fc2c8bf4` · **Flag:** `config.triton.online_softmax_fast_combine`.
**Status:** MERGE-READY via carved prerequisite — `pr/online-softmax-clean`.

## Branch & how made mergeable
`pr/online-softmax-clean` (rooted on `5e2ab3055de`):
- `239a84e2550` — **prerequisite**, carved from mega `97385fb3273`:
  `config.triton.scalar_reduction_accumulators` + the `reduction()` codegen
  rewrite in `codegen/triton.py` (the scalar online-softmax `_block_max` path
  that a26fc patches, carried whole as one interleaved method rewrite).
- `5205038a9ed` — `a26fc2c8bf4` applied. `config.py` merged clean; `triton.py`
  had one **textual-context** conflict in `TritonKernel.__init__` — a26fc's real
  addition there is `self._online_softmax_fast_max` (kept); the diff's
  `has_store_with_rindex` lines were context from an out-of-stack mega-lineage
  commit (absent at base, 0 refs in tree) and were dropped.

## Conflict classification
Mostly REAL SYMBOL DEPENDENCY (the scalar online-softmax codegen path is
mega-introduced) plus one TEXTUAL CONTEXT drift, resolved as above.

## Verification (2026-07-15)
2 touched `.py` files ast-parse; 0 conflict markers; symbols
`online_softmax_fast_combine` (config), `_rewrite_masked_load_other_neg_inf`,
`_online_softmax_fast_max`, and the `DeferredLineBase` import all present
(`DeferredLineBase` is defined in `utils.py` at base); patch applies clean onto
pristine base.
**Compile-smoke (2026-07-15, B200, PYTHONPATH-shadow):** import OK (y) — all the
above symbols resolve at import. Representative repro
`repros/canonical/amax_sum_02064a1e60ac` (softmax) `torch.compile`s to a looped
`triton_red` online-softmax kernel — **GOOD**. Flag A/B (`torch._dynamo.reset()`
between runs): with `online_softmax_fast_combine` **off** the kernel uses
`triton_helpers.max2` (0 fast lines); **on** it uses the native
`tl.max(tl.where(..., -inf))` fast path (1 fast line, 0 `max2`) — the exact
substitution `a26fc2c8bf4` introduces. Numerics vs eager: **EXACT**
(`max_abs=0.0`) in both flag states. **Net: imports + compiles a representative
repro to a numerics-valid result on B200, fast-combine flag path exercised both
ways without crash; full CI not run.**
**Perf verification (2026-07-15, B200, A/B PYTHONPATH-shadow base 5e2ab vs branch
tip, fresh inductor cache per arm, bench_parallel locked path): REPRODUCES
(directionally).** The commit's own named repro `amax_sum_sum_6fd07d12d98a` no
longer exists in the corpus, so the nearest affected family was benched (5 shape
points): branch vs base `amax_sum_02064a1e60ac`/TrOCR 154.5→78.7us (**1.96x**, gap
3.22x→1.64x), `amax_sum_sum_3f000d9caa57` 1.11–1.12x on all 3 points,
`amax_sum_sum_04ddf882ff17` 1.03x — **geomean 1.23x**, every point net-positive
(repeat run matched to <0.1%). Flag isolation (branch with
`TORCHINDUCTOR_ONLINE_SOFTMAX_FAST_COMBINE=0`): the carried prereq scaffolding
alone REGRESSES the amax_sum_sum points 0.90–0.92x vs base; the fast-combine flag
recovers a consistent **1.14–1.22x on every point** (flag-on-vs-off geomean
1.18x), more than paying back the scaffolding cost. The flag defaults ON in the
branch. Caveat for review: if the prereq is upstreamed separately from the
fast-combine commit, the intermediate state is a measured 8–10% regression on this
family. Raw data: `perf_verify/RESULTS.json` (this dir).

## Shared scaffolding note
The prereq's `reduction()` codegen rewrite is the same one PR4 (scalar-acc) needs.
If upstreamed together, merge the two prereqs into one shared scaffolding PR.

## Summary
Replaces the two-op online-softmax combine with a native `tl.max` on the
NaN-identical fast path.

## Measured impact
- **Kernel-geomean +0.83pp / per-model e2e +0.33pp.**
- genai: SoftmaxForward +10%, CrossEntropy +20%.

## Test plan
Softmax/CE genai micros; attention/softmax-bearing models. Flag-gated A/B.
Numerics-gated (NaN-identical path).

written with claude code
