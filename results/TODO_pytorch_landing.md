# TODO — PyTorch upstreaming triage + session close-out

Forward-looking work list (as of 2026-06-19). Status log lives separately in
`OVERNIGHT_PLAN_2026-06-19.md`; this is the "what's left to do" doc.

Provenance pins: perf branch = `tmp_work` @ `daa79cd25ca` (tagged
`perf-snapshot-2026-06-19`); true perf-work base = `5e2ab3055de` (NOT `244fdb379d11`
— that's contaminated by 56 upstream PRs); better-benchmark branch
`investigations-june-2026` @ `e0b93780b`; `main` @ `6e1647b0d` (product landed).

---

## A. THE BIG ONE — PyTorch landing triage (Phase 2). NEEDS USER GO (GPU spend).

The goal: turn the perf branch into a set of justified, individually-correct,
individually-reviewable PRs. The framing "139 commits, not landable" is a mirage
(reverts/churn collapse); the real question is WHICH feature-units carry the
+4.51% geomean and are correct + general enough to upstream.

### A0. Re-decompose into feature-units (CPU, cheap — do FIRST; corrects Phase-1)
Phase-1 said "1 CORE + 11 passes" — **that under-counts.** Verified: the "CORE"
mega-commit `97385fb3273` (5,237 ins, 17 files) is itself a GRAB-BAG — its own
message lists ~6 separate features (BN-split-threshold 6x, MOR Triton finalize
35% Swin, CE gather-into-softmax 0.8%, reduction-chaining) AND bundles 5 brand-new
pass files (scatter_reduce_fusion 1929L, linear_reduction_elimination 726L,
slice_scatter_elision, as_strided_scatter_elision, layout_transform_store_sinking)
plus core simd.py/triton.py/ir.py edits.
- TODO: decompose the mega-commit by FEATURE (not by commit) + the 11 standalone
  passes + loose core-codegen changes → the real unit list is ~16-18 units, not 12.
- Map each unit to: its config flag (if any), the files it owns, its dependencies
  on the core codegen changes (some passes need the mega-commit's ir.py/triton.py
  hooks → not cleanly separable).

### A1. Per-unit MODEL-IMPACT attribution (the GPU campaign)
The mega-commit self-reports per-feature impact ("6x", "35% Swin", "0.8%") but
those are KERNEL-MICROBENCH guesses by the author. The model-extrapolation
pipeline converts them to real per-model geomean impact — THIS is what it was built for.
- Method (recommended, scoped ~16-24 GPU-hr): (1) CPU codegen-diff proxy — for each
  unit, which models' emitted kernels actually change? rank by that; (2) bench only
  the movers via their config flags (the 11 passes are flag-gated → clean A/B;
  the mega-commit features must be benched cumulatively / by selectively reverting).
- Full per-unit walk = ~62 GPU-hr (don't, unless needed).
- Output: each unit tagged with measured model geomean impact. Units below the
  ~0.82% A/A noise floor do NOT justify a standalone PR (drop or bundle).
- KNOWN going in: total branch = median +2.18% / geomean +4.51% (vs 5e2ab). 3
  regressions (pytorch_unet -20%, Longformer -9.4%, pyhpc -12%). Most of +4.51%
  likely concentrated in a few units — find which.

### A2. Per-unit CORRECTNESS + GENERALITY gate
- Each unit: numerics + regression check (we know 3 regress; others may be net-neg/wrong).
- Generality: principled vs corpus-specific hack? (cf. the inline+tile finding —
  suppress-inline is pragmatic, the CSE-hoist is the RIGHT fix). Some "wins" need
  rework before they'd survive review.
- Output per unit: **LAND / BUNDLE / DROP / REWORK** verdict → the PR plan.

### A3. The regression fixes (handoff, NOT our commits)
results/regression_investigation/ has validated patches:
- R1 (pytorch_unet): `R1_inline_profitability_gate_fix.patch` — VALIDATED net-positive
  (flips -19.96% -> +1.87%, headline improves, pass-wins preserved). BUT the deeper/
  better fix is CSE-hoisting the 4x-replayed BN param loads (the real cost), not
  suppress-inline — see R1_tiling_alt findings. Decide which to upstream.
- R3 (Longformer): `R3_diagonal_skew_profitability_gate.patch` — VALIDATED (808->659us
  + fixes an 11-repro cluster; one minor tradeoff). The ASK-1 profitability gate.
- R2 (pyhpc): diffuse, no single fix — needs a global "don't increase kernel count
  on large pointwise graphs" guard. Document, don't patch.
- TODO: package R1/R3 as PRs for the branch owner (their pytorch, not ours).

---

## B. SAFE CLOSE-OUT ITEMS (no big decision; can do anytime)

### B1. Land results/b200/ to main  [capstone repro PASSED — cleared]
- Repro confirmed floors reproduce (oracle median ratio 0.9992, 96.9% within 10%);
  D2 ranking holds; contamination stays gone. So it's safe to publish.
- BUT regenerate the b200 snapshot first with the CORRECTED D1 framing: baseline
  5e2ab (not 244fdb), median +2.18% / geomean +4.51%, + note compile_us run-to-run
  variance (floors stable, individual ratios move). Then rebuild staging onto
  current origin/main (6e1647b0d) + FF-push. Recipe in OVERNIGHT_PLAN §"IF REPRO CONFIRMS".
- HELD for user eyeball (it's a content update to a public artifact, not just a push).

### B2. CURRENT_REPRO_VERSION follow-up → main
- `e0b93780b` (derive version markers from constant + sync stale fallback) is on the
  branch but NOT yet on main (main got the original 2->3 in 9789cda0c, not this followup).
  It was applied to main as `6e1647b0d`... CONFIRM: main already has it (6e1647b0d).
  [If yes, this item is DONE — verify.]

### B3. Stale task list cleanup
- Tasks #8-13 (test-debt plan) are DONE (suite green 329/0, committed). Mark complete /
  clear the task list.

---

## C. OPEN QUESTIONS FOR USER
1. Phase 2 scope: full per-unit walk (~62 GPU-hr) vs scoped proxy-then-bench
   (~16-24) vs CPU-proxy-only-first? (Recommend: CPU proxy first, then targeted bench.)
2. R1 fix to upstream: the validated suppress-inline patch, or invest in the
   "correct" CSE-hoist fix (keeps the fusion, likely beats baseline)?
3. results/b200 → main: regenerate with corrected 5e2ab framing then push? (B1)
4. How much of this is OURS to upstream vs hand to the branch owner? (the perf
   branch is theirs; we have the triage + validated fixes.)

---

## D. DON'T (standing constraints)
- No PRs / pushes to the pytorch remote without explicit go.
- No Phase-2 GPU walk without scope sign-off.
- No reset --hard / history rewrite on shared branches; never touch tmp_work WIP.
- Stage artifacts under /tmp/scratch_space (durable), never bare /tmp.
- Drive bench_parallel/bench_oracle + post-process; never write new timing code.
