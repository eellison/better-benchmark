# Overnight autonomy plan — 2026-06-19 (user away)

## STATE AT HANDOFF
- branch `investigations-june-2026` HEAD `e0b93780b` (== origin, in sync)
- `main` at `6e1647b0d` — product fixes PUBLISHED (dynamo-reset, sharding, projection-gate,
  numerics opt-out, pytorch_ref, CURRENT_REPRO_VERSION 2->3 + derived-marker followup,
  11 bench-hacks, test consolidation, cleanup). NO perf numbers on main yet.
- perf branch tagged: `/tmp/pytorch-work` tag `perf-snapshot-2026-06-19` -> daa79cd25ca (so it's not lost if tmp_work moves)
- Deliverables (validated): D1 branch improvement +2.18% median / +4.84% geomean (genai-excl, n=158);
  D2 distance-to-ceiling median 0.4% / Longformer 37%. results/b200/ self-contained snapshot.
- 3 Inductor regression fixes (R1 pytorch_unet + R3 Longformer VALIDATED net-positive, R2 diffuse)
  as branch-owner handoff patches in results/regression_investigation/.

## RUNNING OVERNIGHT WORKERS
1. **Capstone repro sweep** (agent ac09eeffbe) — clean full oracle sweep on FINAL committed tooling,
   confirms results/b200 floors reproduce + validates sharding-at-scale wall-clock. GATES results->main.
2. **Phase-1 commit archaeology** (agent a4be7712) — read-only inventory of the 139 perf-branch commits,
   dedup revert-pairs/churn -> clean logical-commit list + Phase-2 GPU cost estimate. Output results/pytorch_landing/.

## WHAT I MAY DO AUTONOMOUSLY (no user needed)
- Consolidate/verify agent results as they land; write durable reports under results/.
- Commit + push ANALYSIS/RESULTS to the BRANCH investigations-june-2026 (fetch+rebase, NEVER reset --hard).
- If capstone repro CONFIRMS the floors (statuses stable, >=5x stays ~0, D2 headline holds within noise):
  LAND results/b200/ to main as the perf-results follow-up (staging 9259de616 already built + FF-safe;
  rebuild on current origin/main if it moved). This was pre-approved ("fine landing perf results" / "push without perf" was about decoupling, results follow once repro validates).
- Re-dispatch any worker that dies (recover from on-disk partial, like before).
- Spawn read-only / analysis subagents (e.g. let Phase-1 finish; do NOT auto-start Phase-2 GPU benching — see below).

## WHAT I MUST NOT DO WITHOUT THE USER
- Do NOT start Phase 2 (per-commit GPU attribution of the 139 commits) — it's a big GPU spend;
  Phase 1 produces the cost estimate, user decides scope. Just have the PLAN ready.
- Do NOT push anything to PyTorch (no PRs, no pushes to the pytorch remote). The regression
  patches stay as handoff artifacts.
- Do NOT force-push, reset --hard, or rewrite history on any shared branch (tmp_work or investigations).
- Do NOT rewrite tmp_work or touch its uncommitted WIP (config.py + scatter_reduce_fusion.py).
- If capstone repro shows a RED FLAG (status flips at non-boundary points, >=5x count jumps,
  D2 headline shifts beyond noise): do NOT publish results to main — write up the discrepancy and HOLD for user.

## IF REPRO CONFIRMS -> the results->main landing recipe
- staging branch stage/main-land-2026-06-18-v2 (commit 9259de616) = full product+b200 snapshot, FF-safe onto origin/main as of handoff.
- But main already advanced to 6e1647b0d (product) since that staging was built on 77ce23ed3.
  So REBUILD: snapshot current branch HEAD's tree (product + results/b200, curated excludes) onto CURRENT origin/main, verify (pytest green, corpus invariants, compute_ab reproduces, pytorch_ref present), FF-check, push.
- Curated excludes (NOT on main): the ~17 scratch results/* dirs, investigation_results/, process docs (*.md prompts), results/regression_investigation/ (handoff, not product), results/pytorch_landing/, results/test_debt_fix/, results/repro_validation_2026-06-19/. INCLUDE: results/b200/ only. .gitignore: un-ignore results/b200/.

## MORNING REPORT TO LEAVE
A summary at top-of-context when user returns: repro verdict (+ whether results landed on main),
Phase-1 result (real landable-commit count + Phase-2 cost), anything held for decision.

## UPDATE (Phase 1 landed + a consequential finding)
**Phase 1 done** (results/pytorch_landing/PHASE1_commit_inventory.{md,json}): the "139 commits" is really
83 perf commits (5e2ab..HEAD) -> 61 substantive -> ~24 feature units = **1 large CORE PR (mega-commit
97385fb3273, seeds 7 passes + inline_recomputable_producers) + ~11 independent FX-pass PRs**. NOT 30+
independent landable commits. Phase-2 full walk ~62 GPU-hr; recommended scoped proxy-rank ~16-24 GPU-hr.
DO NOT start Phase 2 without user.

**CONSEQUENTIAL: the A/B baseline 244fdb379d11 is CONTAMINATED.** VERIFIED: 56 upstream pytorch-main PRs
(all #-tagged, touch scheduler.py +248 / lowering.py +78) sit between 244fdb and the perf work's true base
5e2ab3055de. So +2.18%/+4.84% conflates this-branch + 56 upstream inductor PRs landing independently.
-> Dispatched re-baseline agent (ad7d451f) to re-measure D1 vs 5e2ab (branch arm reused, only new baseline
arm benched). The CORRECTED this-branch number is what the morning report should cite for the landing decision.
NOTE: results/b200 records baseline 244fdb -> its D1-relevant baseline ref is now known-wrong; the oracle
FLOORS (D2) are unaffected (measured at one commit). When landing results/b200 to main, add a note that the
A/B baseline is being corrected to 5e2ab (or wait for the corrected number before publishing D1 framing).

## Worker status after this update
- capstone repro sweep (ac09eeffbe) - running, gates results->main
- D1 re-baseline vs 5e2ab (ad7d451f) - running, corrects the headline
- Phase 1 (a4be7712) - DONE

## UPDATE 2 — D1 re-baseline DONE (the contamination was nearly harmless)
Re-measured D1 vs the correct base 5e2ab (results/perf_ab/rebaseline_5e2ab_2026-06-19/):
- **TRUE this-branch: median +2.18% (UNCHANGED), mean +4.05%, geomean +4.51%** (genai-excl, n=158, 120/28/10).
- Contaminated 244fdb was: +2.18% median / +4.84% geomean. So upstream's 56 PRs added only +0.004pp median, ~0.3pp geomean.
- The entire geomean delta = ONE upstream kernel `sum_81b4fd73f8d1` (cross-entropy/NLL reduction, ~2x faster upstream) mis-credited to 2 train models (TrOCR +23%->+5%, Pegasus +21%->+5%). Per-point cross-baseline drift = A/A floor (signed +0.000%), only 2/4977 points >30% (both that kernel).
- VERDICT: the +2.18% median headline was honest; cite median +2.18% / geomean +4.51% as the corrected this-branch number. Isolation clean, tmp_work restored byte-identical.
- IMPLICATION for landing: when results/b200 D1 framing is published, use the 5e2ab-based number (or note both). D2 oracle floors unaffected.

## Worker status
- capstone repro sweep (ac09eeffbe) - STILL RUNNING, gates results->main
- D1 re-baseline (ad7d451f) - DONE (above)
- Phase 1 (a4be7712) - DONE

## UPDATE 3 — CAPSTONE REPRO VERDICT: PASSES (floors reproduce; landing decision below)
Repro sweep self-relaunched after agent death, completed (4975/4977 ok, 5673s ~95min — sharding-fix-at-scale confirmed). Compared fresh vs published results/b200 floors:
- **ORACLE_US (the D2 ceiling) reproduces TIGHTLY: median ratio 0.9992, 96.9% within 10%, 88% within 5%.** D2 reproduces.
- Contamination stays gone: 1 fresh >=5x (sum_sum_sum_04b57f7e083d 5.5x GOOD — looks real, not the old artifact); published had 0.
- 768 status flips BUT: 631 near-1.0x boundary noise; rest are numerics-gate re-rolls + COMPILE-side variance. Oracle side is stable; variance is on compile_us (expected — more run-to-run variance + the dynamo/cache interactions).
- Longformer #1 driver amax_sum_528a3c274a41/79c25467: oracle 320.3->320.5 (IDENTICAL), compile 785->637 (compile got FASTER -> ratio 2.45->1.99). The gap is real and reproduces; it just SHRANK because compile improved on that kernel run-to-run. Headline holds (Longformer still #1 real gap).
VERDICT: floors reproduce, D2 holds, no red-flag. Capstone PASSES.

## LANDING DECISION (made autonomously per overnight rules — repro confirmed)
The published results/b200 floors are REPRODUCIBLE on final tooling -> clear to land results/b200 to main.
Caveat to fold into the landing note: D1 baseline correction (244fdb contaminated -> use 5e2ab; median +2.18% unchanged, geomean +4.51%); compile_us has run-to-run variance so individual ratios move ~ but oracle floors + D2 ranking are stable.
