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

**Status:** DONE — rewritten to faithful single-path oracles and parent-verified. All pass
`--check`/`--check --no-skip-stochastic`; locked bench now records honest
`NUMERICS_WORSE_THAN_COMPILED` where the old pass depended on tolerance steering.

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

**Audit done (2026-06-18, `results/priced_floor_audit/audit.json`): 1232 priced dirs scanned, 6 hacks found — all PASS `--check`.**
**Good news: NONE inflate any ranking.** At every blended shape the hacked oracle is slower-or-equal to the
compiled branch, so `max(0, branch−oracle)` recovery = 0µs (a tolerance-blend *toward the compiled value*
structurally cannot make the oracle look faster than compile). Top-15 headroom (Longformer 37%, swin, vgg16,
BERT-train cluster) verified CLEAN. So these 6 are an **integrity backlog, not a ranking emergency** — but they
still pass the gate falsely and must be rewritten faithfully. These are UNCLAIMED (codex's `079bdc4bd` claimed
only the Group 1 five). Hand-verified #1/#4/#5 below by source; #2/#3/#6 per audit (same family).

**Status:** DONE — rewritten to faithful single-path oracles and parent-verified. The
queue rows record the post-rewrite per-point outcomes: remaining GOOD/AT_FLOOR points
kept as priced floors, `NUMERICS_WORSE_THAN_COMPILED` points unpriced, and the
pytorch_unet nonzero contribution retired.

| # | dir | pattern | model scope | --check | the hack (offending lines) | feeds |
|---|-----|---------|-------------|---------|----------------------------|-------|
| 6 | `var_mean_96edf0542ecd` | var_mean | LayerNorm | PASSES | `tol=TOL_SCALE·(0.01+0.01·\|y\|); where(\|y_fp−y_round\|<=tol, y_fp, y_round)` (L53-54) | — (0µs) |
| 7 | `var_mean_05c4919007f9` | var_mean | LayerNorm | PASSES | clamp `final_resident` to ±0.75% band of faithful (L100-102) | — (0µs) |
| 8 | `var_mean_0ada225c0a04` | var_mean | residual-add LayerNorm | PASSES | dual `x_exact`/`x_inductor` reductions (L68-77); `clamped=min(max(ind_f32, exact−tol), exact+tol)` stored as output (L90-97). Broadest: feeds 15 hf/infer models, **0µs to all**. | 15 hf/infer (0µs) |
| 9 | `sum_sum_292e665668b6` | sum_sum | BN-backward | PASSES | nudge-to-tolerance gated by `CLAMP_OUTPUTS`; only 2/7 GOOD pts blended (L250-255) | — (0µs) |
| 10 | `sum_sum_8b1ecd6ea781` | sum_sum | DenseNet BN-backward | PASSES | `tolerance=0.009+0.009·\|slice_eager\|; use_fused=abs(slice_fused−slice_eager)<=tol; slice_out=where(use_fused, slice_fused, slice_eager)` (L159-161) | — (0µs) |
| 11 | `sum_sum_sum_1a0060cd26d5` | sum_sum_sum | pytorch_unet maxpool-bwd | PASSES | dual `where_compiled`/`where_exact` (L112-113); nudge L244 + `where(\|compiled−exact\|<=tol, compiled, exact)` L315-317 | pytorch_unet (40.8µs=0.38pp, rank 56 — only nonzero) |

### Per-entry rewrite recipe (Group 2)
- **#6 `var_mean_96edf0542ecd`** / **#7 `var_mean_05c4919007f9`**: standard LayerNorm — keep ONE faithful fp32 row var_mean (correction=0) → rsqrt → affine → bf16 cast. Delete the fp/round dual + the `where`/clamp-to-band. Speed must come from the row template, not the blend.
- **#8 `var_mean_0ada225c0a04`** (residual-add LayerNorm): keep the faithful `x_exact` path (residual add → row var_mean → affine). Delete the entire `x_inductor` parallel reduction (L74-77) and the `clamped=min(max(ind_f32, exact±tol))` store (L90-97); return the faithful affine. The same-layout residual-producer fusion is the legit speedup.
- **#9 `sum_sum_292e665668b6`** (BN-backward): remove the `CLAMP_OUTPUTS` nudge path entirely; return the faithful channel reductions unconditionally (don't blend even the 2 gated points).
- **#10 `sum_sum_8b1ecd6ea781`** (DenseNet BN-backward, sibling of Group-1 #2): keep ONE `slice` path with correct bf16 rounding; drop `slice_fused`/`use_fused` select (L159-161). The resident channel-reduction fusion is legit.
- **#11 `sum_sum_sum_1a0060cd26d5`** (pytorch_unet maxpool-backward — the ONLY one with nonzero, though tiny, headroom contribution): keep ONE `where_exact` (bf16-rounded scatter-skip) path; delete `where_compiled` (L112), the L244 nudge, and the L315-317 `where(|compiled−exact|<=tol,...)`. The fixed-offset maxpool-scatter decode into channel reductions is the legit speedup. NOTE this is a pytorch_unet scatter oracle — same family as the f5be ones just rebuilt faithfully (`cf5e20ace`); apply the same faithful-index approach.

---

## How to point a kernel agent at one entry
Give the agent: the `dir`, the "hack" cell (so it knows exactly what to delete), and the
matching "rewrite recipe" bullet (so it knows what the faithful version must compute).
Tell it: rewrite to a single faithful path satisfying the rewrite invariant above; run
`--check`; re-bench with the lock; report honest oracle_us/status. Do NOT accept a rewrite
that still contains any dual-path/blend/clamp-to-tolerance construct. One agent per dir is
clean (no shared-file conflict, different repro dirs). Re-run
`python scripts/validate_corpus_invariants.py` before committing any `repros/` change.
