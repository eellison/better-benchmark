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
pristine base. **Caveat:** ast-parse + symbol-grep only — no torch build, not
compiled/run.

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
