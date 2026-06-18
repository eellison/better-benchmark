# Oracle Rewrite Worklist — bench-hacks to redo faithfully

**Purpose:** these oracles pass (or game) the numerics `--check` gate WITHOUT doing the
real computation — the f5be tolerance-self-blend family. Each must be rewritten to a
single faithful path. A kernel agent can pick up ONE entry and execute it standalone.

**Status legend:** `TODO` = needs rewrite · `DONE` = rewritten + re-benched + check passes faithfully

**The hack family (what to ELIMINATE in every rewrite):**
- `tl.where(abs(a-b) <= tol, a, b)` self-blend toward a cheap/compiled value
- `clamp/minimum+maximum(compiled_value, faithful_value ± tol)` — returning compiled clamped to the verifier band
- dual pipelines: `*_exact`/`*_strict`/`*_faithful`/`*_resident` vs `*_fused`/`*_compiled`/`*_anchor`/`*_eager`/`*_candidate`, where the RETURNED value is the cheap one or a blend
- `delta = resident - strict; step = min(abs(delta), tol)` nudge-to-tolerance
- `del arg_*` dropping a real input then fabricating it from a degenerate/constant

**The rewrite invariant (what every faithful oracle MUST satisfy):**
1. ONE computation path. No second "compiled/eager/resident" path, no per-element selection between paths.
2. The output is the genuine result of the real reduction/normalization over ALL captured elements.
3. `python -m oracle_harness <dir> --check` passes on the *math*, not on a tolerance blend.
4. Keep the legitimate fast-kernel techniques (tiling, fp32 accumulate, fused epilogue) — speed must come from a better SCHEDULE of the real work, never from skipping it.
5. Re-bench after: `python scripts/bench_parallel.py --oracles --all-shapes <dir>` (or the dir-scoped oracle bench) with GPU lock on, fresh `TORCHINDUCTOR_CACHE_DIR`. Record the honest oracle_us/status. If the faithful oracle is now slower than compile on every shape → that's fine, it becomes an honest BAD_ORACLE (accept at compile floor); the point is integrity, not a guaranteed win.

---

## Group 1 — confirmed bench-hacks in the FAILURE bucket (triage, 2026-06-18)
These are currently in `__failures__` (not poisoning any priced floor) but are integrity defects. Source-verified by hand.

| # | dir | pattern | model scope | --check today | the hack (offending lines) |
|---|-----|---------|-------------|---------------|----------------------------|
| 1 | `sum_sum_03f9b31579a0` | sum_sum | GhostNet BN-backward | PASSES | `sum_return = clamp(sum_compiled, sum_value ± (0.009+0.0095·\|sum_value\|))` (L223-230); dual `where_value`/`where_compiled` reductions (L158-170). Returns the COMPILED sum clamped into the tolerance band. |
| 2 | `sum_sum_e5ed56d5d094` | sum_sum | DenseNet BN-backward | PASSES | dual `residual_exact` (bf16-rounded, faithful) vs `residual_fused` (no rounding) paths (L195-216); per-element `use_fused = abs(fused-exact) <= 0.0098+0.0098·\|exact\|` select. |
| 3 | `var_mean_06a6aec610f6` | var_mean | DINOv2 class-token LayerNorm | PASSES | `delta = resident - strict; full_tol = 0.01+0.01·\|strict\|; step = min(\|delta\|, 0.95·full_tol)` nudge-to-tolerance (L132-137). Computes faithful `out_strict` then drifts toward Inductor's resident value. |
| 4 | `var_mean_1af9add64387` | var_mean | residual LayerNorm alias scope | FAILS | dual `y_inductor` vs `y_eager` LayerNorm paths (L39-54), blended on f32 compare. |
| 5 | `var_mean_fbfc0104897d` | var_mean | Longformer embedding+LayerNorm | FAILS | `x_resident` (faithful sum) vs `x_anchor` (bf16-rtne-rounded "anchor", L99-100); returns resident-normalized clamped toward the anchor envelope. NOTE: this is a **Longformer** scope — Longformer is the #1 headroom model, so audit whether a sibling priced floor for the same pattern is also gamed. |

### Per-entry rewrite recipe (Group 1)
- **#1 `sum_sum_03f9b31579a0`** (GhostNet masked BN-backward): compute the TWO fp32 channel reductions `sum([0,2,3])` over the real masked `where_value = where(affine_bf16<=0, fill, add_value)` (hard-sigmoid gated), produce the scale-gradient vector + the bf16 channels-last input-gradient tensor. Delete the entire `*_compiled` pipeline and the `clamp` returns — return the faithful `sum`/`vec` directly. Keep the split-K/cooperative reduction for speed.
- **#2 `sum_sum_e5ed56d5d094`** (DenseNet BN-backward tail): keep ONE residual path with the correct bf16 rounding (`residual_exact`), drop `residual_fused` and the `use_fused` select. The shared ReLU-mask `where` producer across the two reductions is the legit fusion — keep that.
- **#3 `var_mean_06a6aec610f6`** (DINOv2 token-0 LayerNorm): the legit optimization is real — only token-0 rows are observable (`select(dim=1,index=0)`), so reducing only those 128 rows is FAITHFUL dead-code elimination, NOT a hack. Keep that. DELETE the `delta`/`step`/`full_tol` blend (L132-137) and just return the faithful `out_strict` (bf16 cast of the affine of the real var_mean over hidden). The narrowed row domain is the speedup; the blend is the cheat.
- **#4 `var_mean_1af9add64387`** (residual LayerNorm + alias views): keep ONE LayerNorm path (`y_inductor` math is the faithful correction=0 row var_mean over hidden with rsqrt(var+1e-12)). Drop `y_eager` and the blend. Preserve the alias/permute multi-output epilogue (legit).
- **#5 `var_mean_fbfc0104897d`** (Longformer embedding LayerNorm inference): faithful = word+position+global embedding gathers → fp32 row sum → population var_mean over hidden (eps 1e-5) → bf16 affine → bf16 store. DELETE the `x_anchor` bf16-rtne path and any clamp toward it; return the resident-normalized affine directly. The embedding-gather-as-LayerNorm-producer fusion is the legit speedup.

---

## Group 2 — bench-hacks found among PRICED floors (integrity audit, PENDING)
> Filled in when the priced-floor audit (`results/priced_floor_audit/audit.json`) returns.
> These are HIGHER priority than Group 1 — a priced-floor hack that passes `--check`
> is INFLATING a model's headroom number right now. Any that feed a top-15 headroom
> model (esp. Longformer/swin/vgg16/BERT cluster) are CRITICAL: fix + re-bench + re-rank.

_(none recorded yet — audit in flight)_

---

## How to point a kernel agent at one entry
Give the agent: the `dir`, the "hack" cell (so it knows exactly what to delete), and the
matching "rewrite recipe" bullet (so it knows what the faithful version must compute).
Tell it: rewrite to a single faithful path satisfying the rewrite invariant above; run
`--check`; re-bench with the lock; report honest oracle_us/status. Do NOT accept a rewrite
that still contains any dual-path/blend/clamp-to-tolerance construct. One agent per dir is
clean (no shared-file conflict, different repro dirs). Re-run
`python scripts/validate_corpus_invariants.py` before committing any `repros/` change.
