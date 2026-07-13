# PHASE-2 STEP A2 — Ranked PR Verdict Table

**Date:** 2026-06-23
**Mode:** CPU analysis off existing data (NO benching, NO GPU).
**Inputs:** `A1_commit_walk/FULL_per_commit_table.txt` (per-commit geomean deltas, 5e2ab..HEAD),
`A0_feature_units.json` (commit→feature→flag map), `regression_investigation/` (R1/R2/R3 + validated patches),
`A1_commit_walk/walk_analysis.json` (coarse 6-state cumulative + per-model xcheck trajectories).

> **Honest caveats baked into every verdict below.**
> 1. **rsqrt #1 (+5.64pp) is the single largest mover but is PENDING kernel-verification.** As of this writing
>    `results/pytorch_landing/rsqrt_verify/REPORT.md` **does NOT exist** → rsqrt is marked **VERIFY-FIRST**
>    (LAND-pending). Do NOT finalize as LAND until that report confirms a fusion-unlock mechanism.
> 2. The mega-commit `97385fb3273` (+1.95pp) is a **BUNDLE of ~9 A0 units** — it is *not* one PR. Its delta is
>    flagged **DECOMPOSE-LATER**; do not treat the +1.95pp as attributable to any single feature.
> 3. `compile_us` / geomean has run-to-run variance; the A1 walk's own **noise floor is ±0.82pp**. Any single
>    delta at or below ~0.82pp is at/near noise and is treated as BUNDLE-or-noise, never standalone-LAND.

---

## 1. Ranked verdict table (by measured |Δpp|)

Δpp = geomean delta vs previous benched commit, from `FULL_per_commit_table.txt`. Flag/file from `A0_feature_units.json`.

| Rank | Commit SHA | A0 unit | Feature | Δpp | Flag / file | Verdict | Rationale |
|------|-----------|---------|---------|-----|-------------|---------|-----------|
| 1 | `a73d1583b34` | U20 | rsqrt_canonicalization `reciprocal(sqrt(x))→rsqrt` | **+5.64** | `config.rsqrt_canonicalization` (default 1); `post_grad.py` ~L2166 | **VERIFY-FIRST** (LAND-pending) | Largest single mover, own flag, clean independent post_grad rewrite → would be the headline PR. BUT kernel-verify report absent; +5.64pp is implausibly large for a pure canonicalization unless it unlocks a downstream fusion. Confirm mechanism, then upgrade to LAND. If refuted, downgrade (suspect bench artifact). |
| 2 | `97385fb3273` | U01,U07,U08,U09,U10,U25,U30,U31,U33 (+ dead reduction_chaining) | mega-commit "All kernel optimization improvements" | **+1.95** | many flags — see A0 | **DECOMPOSED ✓ (see §6)** | Grab-bag of ~9 units; flag-ablation (§6, 2026-06-23) split the 6 flag-ablatable units: only U10 (layout-sinking) is a clean LAND; U01 corpus-dead; U07/U08 at-floor BUNDLE; **U09 + U25 NET-NEGATIVE** (U25 = the pytorch_unet -0.50% source). The +1.95pp is NOT evenly spread — bulk is in the 3 NO-FLAG units (U30 MOR/Swin, U31 CE, U33 BN) needing selective-revert. **Do NOT land the flags as one bundle.** |
| 3 | `ab29345d82d` | U31 | CE loop-invariant hoisting (realize gather / target_logit hoist) | **+1.22** | NO flag — selective revert (`inductor_prims.py`, `ops_handler.py`, `lowering.py` L9177, `triton.py`) | **LAND** (as CE PR) | Real, well-above-noise (+1.22 vs 0.82 floor), self-contained CE-online-softmax improvement. No flag → ships as a focused selective-revert PR carrying the `online_softmax_cross_entropy` prim + the hoist. Mechanism is principled (hoist loop-invariant target logit out of the per-row reduction). |
| 4 | `a26fc2c8bf4` | U28 | online-softmax fast combine (native tl.max / NaN-identical fast path) | **+0.83** | `config.triton.online_softmax_fast_combine` | **LAND** | Above noise, own flag, principled (replaces 2-op combine with native max on the NaN-identical path). Independently PR-able even though it lives in shared `triton.py` (L4622/5159/5304). |
| 5 | `1a824e0747a` | (loose / U25-adjacent) | realize_reads_threshold 4→30 | **+0.83** | `config.realize_reads_threshold` scalar | **DROP (MOOT)** | Cancelled by its own revert `f74a7f13723` (−0.82pp) → net ≈ 0. Do NOT land the raise. (The memory.py +28 from the revert commit survives and is attached to U25, not this threshold.) |
| 6 | `f595e0cac3a` | U06 | topological-ordering fix in select_scatter_sparsity | **+0.50** | `config.select_scatter_sparsity` (default 1) | **BUNDLE** (into U06 select_scatter_sparsity PR) | Below the 0.82 noise floor on its own, but a genuine *correctness/ordering* fix for the select_scatter_sparsity pass. Bundle as the fix commit inside the U06 pass PR rather than standalone. |
| 7 | `ca8f961d6b0` | U29 | scalar_acc gate on `num_load ≤ 4` | **+0.59** | `config.triton.scalar_reduction_accumulators` + `config.scalar_acc_configs_without_cd` | **LAND** | Below floor in raw pp but this commit is a **regression-fixing gate** (fixed measured 14–22% regressions from the ungated scalar_acc); the net is positive and it's principled (only use scalar accumulators when load count is small). Own flags. PR-able as the scalar_acc family with its gate. |
| 8 | `9dde2c59a51` | U29 | ungate scalar-acc configs from CD + raise rnumel ceiling | +0.44 | `config.scalar_acc_configs_without_cd` | **BUNDLE** (into U29 scalar_acc PR) | Same unit as #7; the ungate-from-CD + rnumel ceiling are the companion change. Ship together with `ca8f961d6b0`. |
| 9 | `1406552b9d3` | U15 | rotate_half_gather (RoPE rotation → single-load gather) | +0.37 | `config.rotate_half_gather` (default 1) | **BUNDLE / LAND-small** | Below floor; clean independent fx_pass with own flag and a clear mechanism (RoPE rotate-half as one gather). Land as a small standalone PR or bundle with the other independent fx_passes; low risk. |
| 10 | `f0b54e1aa1f` | (revert) | Revert "Tune num_stages for memory-bound pointwise" | +0.35 | n/a (revert) | **DROP-as-pair (net zero)** | This is the *revert* arm of an added+reverted pair (num_stages tune `caa0…`→`f0b5…`); net ≈ 0. Nothing to land; the revert is already the right end-state. |
| 11 | `4427bfd553c` | U03 | scatter_add_into_fusion — dtype-cast variant | +0.35 | `config.scatter_add_into_fusion(_chained)` python bool | **BUNDLE** (into U03 scatter_add_into_fusion PR) | Below floor; extends an existing fusion to the dtype-cast case. Ship inside the scatter_reduce_fusion family PR (depends on the mega-commit's `scatter_reduce_fusion.py` file). |
| 12 | `a85d79a900a` | U23 | segment-aligned split-reduction introduction | +0.25 | `config.segment_aligned_split_reductions` (default 1) | **BUNDLE** (split-reduction family PR) | Below floor; co-evolves with U22/U24 in choices.py/ir.py. Bundle the split-reduction cost-model family into one PR. |
| 13 | `d8e9914094a` | U11 | diagonal_skew_elimination (Longformer scatter read-bypass) | +0.18 | `config.diagonal_skew_elimination` (default 1) | **REWORK** (R3 fix required) | Self-reported "8→5 kernels" but on the actual Longformer band it is the **R3 −9.40% regression** (commits bypasses while the band scatter stays live → blocks fusion). Real win class exists but as-is it regresses the #1-headroom model. **Apply R3 patch before landing** (`R3_diagonal_skew_profitability_gate.patch`, validated). |
| 14 | `d2cd52d3117` | U33 | raise persistent-reduction INNER threshold 1024→8192 | +0.14 | `config.persistent_reduction_threshold_inner` scalar | **DROP (MOOT)** | Added here, **reverted** by `62f27911a59` (−0.24pp). Net negative/zero → do not land the inner-threshold raise. (The *other* U33 Blackwell-BN threshold + low-warp configs are separate and live in the mega-commit.) |
| 15 | `afc3d5956d5` | U16 + U24 | index_through_norm + split-for-fusion + bounds-elide | +0.11 | `config.index_through_norm`, `config.split_reductions_for_fusion`, `config.elide_constant_index_asserts` | **BUNDLE** | Tiny, multi-feature commit. Split: index_through_norm → independent fx_pass PR (own flag); split-for-fusion → split-reduction family PR (U24); bounds-elide → micro-cluster with clamp_embedding. None standalone-worthy. |
| 16 | `56959375c33` | U05 | structured_scatter_decode (affine-iota overlap-add) | +0.13 | `config.structured_scatter_decode` (default 1) | **BUNDLE / LAND-small** | Below floor; clean independent fx_pass. Land in the independent-fx-pass batch. |
| 17 | `b1e8a30af15` | U26 | materialize multi-user transcendentals on reuse | +0.05 | `config.realize_heavy_transcendentals_on_reuse` + `..._indirect_indexing...` python bool | **BUNDLE** (realize-on-reuse family) | Raw +0.05 here, but the coarse-walk cumulative jump at this state is large (this is the realize-on-reuse family). Bundle with U25 realize/scheduler family; note the realize-on-reuse policy is entangled with the R1 regression class. |
| 18 | `86c5451e197` | U17 | BN-inference affine folding FX pass | **−0.76** | `config.fold_bn_affine` (default 1) | **DROP (NET-NEGATIVE)** | Net −0.76pp on the corpus; A0 microbench was already weak (1.27x→1.23x). Do not land; default should be OFF if kept at all. |
| — | `97385fb3273` (sub) | — | reduction_chaining.py (216L) | 0 | `config.triton.reduction_chaining` (missing from config diff → defaults False) | **DROP (DEAD CODE)** | Orphan: never imported/called at HEAD; its config gate isn't even in the config.py diff. Contributes ZERO. Remove from any landing. |

Other small positives (`f9493caabcd −0.10`, `94ad050dee0 −0.05`, `60dd3839913 −0.10`, `daa79cd25ca −0.10`, `e9a67c98a8c +0.06`, `bdc289b3644 −0.14`, `d95f59fb1bc −0.07`, `44802547aa9 −0.26`, `5489b8c2bb9 −0.16`, `b1e8a30af15 +0.05`) are all **within the ±0.82pp noise floor** → BUNDLE into their respective family PRs (dedupe_graph_outputs U18, pointwise_cat heuristic, evict_first U27, MOR-split U30/U33) or treat as noise. None are standalone-LAND.

---

## 2. Net accounting

- **Branch total:** the A1 walk reports the full-corpus geomean at roughly **+4.5%** vs base `5e2ab3055de`
  (the coarse 6-state cumulative `walk_analysis.json` lands at geomean **+3.93%** at the U11 state; the
  finer 37-state FULL table is consistent with the ~+4.51% headline once rsqrt + the mega-commit are included).
- **Do the LAND-verdict deltas sum toward the total?** *Roughly, but the sum is dominated by two deferred/pending items and is NOT cleanly additive:*
  - **rsqrt +5.64pp (VERIFY-FIRST)** alone **exceeds** the entire +4.51% branch total. This is the strongest signal
    that the deltas are **non-additive** (later commits' deltas are measured against a baseline already lifted by
    earlier ones; a single +5.64pp arm can over-count because subsequent reverts/regressions claw some back).
  - **mega-commit +1.95pp (DECOMPOSE-LATER)** is the second-largest contributor and is itself a bundle of ~9 units.
  - The clearly-LAND independent positives — **CE-hoist +1.22, online-softmax +0.83, scalar_acc-gate +0.59,
    topo-fix +0.50** — sum to **≈ +3.14pp**. Adding the small BUNDLE positives (rotate_half +0.37,
    scatter-add-cast +0.35, segment-split +0.25, structured-scatter +0.13, …) brings the
    *non-rsqrt, non-mega* landable set to **≈ +4pp of gross positive**, against which the DROP/MOOT/negative
    items (BN-fold −0.76, threshold raise/revert ≈ 0, inner-threshold raise/revert ≈ 0) net out.
  - **Conclusion:** the LAND set is directionally consistent with a ~+4–4.5% branch win, but **you cannot just add
    the pp** — they overlap (shared cumulative baseline) and the two biggest contributors (rsqrt, mega-commit) are
    not yet finalized. Treat rsqrt as the make-or-break headline: if it verifies, the branch story is "one big
    canonicalization win + a handful of clean fusion PRs"; if it refutes, the branch is "the mega-commit + ~+3pp of
    small clean PRs," and the +4.51% headline needs re-measurement.

---

## 3. DROP list (with reasons)

| Item | A0 / commit | Reason |
|------|-------------|--------|
| BN-inference affine folding | U17 / `86c5451e197` | **Net-negative** −0.76pp on corpus; weak microbench. |
| reduction_chaining.py (216L) | seeded by `97385fb3273` | **Dead code** — never imported/called; config gate absent from diff; ZERO contribution. |
| realize_reads_threshold 4→30 | `1a824e0747a` (+0.83) / reverted by `f74a7f13723` (−0.82) | **Moot** — cancelled by its own revert; net ≈ 0. |
| persistent-reduction INNER threshold 1024→8192 | `d2cd52d3117` (+0.14) / reverted by `62f27911a59` (−0.24) | **Moot/net-neg** — added then reverted; net ≤ 0. |
| num_stages tune for memory-bound pointwise | added `caa0…` / reverted `f0b54e1aa1f` (+0.35 = the revert) | **Net-zero pair** — already reverted to the correct end-state. |
| Degenerate-dropout-elimination pass | added+reverted twice (A0) | **Net zero** — not a unit. |
| Smart realize_hint broadcast-dominated | added `9ae68` / reverted `282760` (A0) | **Net zero**. |
| reduction_chaining sibling: `hoist_invariant_compute` | `52d4cadfac0`, `config...` default **OFF** | **Zero at HEAD** — default off; not landing-relevant unless turned on (and A0 notes "neutral on B200"). |
| `R1_tiling_alt`, `reduction_chaining.py` per the prompt | regression_investigation | reduction_chaining.py confirmed dead-code drop (consistent with A0). |

---

## 4. Regressions to fix before/alongside landing

The branch introduced 3 notable model regressions (A/B base `244fdb37` vs branch `daa79cd2`, 4× B200). Source:
`results/regression_investigation/REGRESSION_HANDOFF.md` + `R{1,2,3}_results.json`.

| ID | Model | Regression | Root cause (A0 unit) | Fix status |
|----|-------|------------|----------------------|------------|
| **R1** | pytorch_unet | **−19.97%** (largest) | U25 `inline_recomputable_producers` inlines a full-size BN+ReLU producer into a 4-tap bilinear gather → recomputes BN at all 4 taps (raw input gathered 157M vs 9.8M). `_is_inline_profitable` prices only FLOPs, omits replayed-input HBM traffic. | **VALIDATED FIX** — `R1_inline_profitability_gate_fix.patch`. Adds replayed-input-traffic term to `_is_inline_profitable`. Recovers R1A→221us, R1B→122us (= flag-off), KEEPS flag on, PRESERVES wins (b0a24575 374→167.5us still inlines). Numerics bf16-rounding only. **Confidence HIGH.** Land alongside any U25 realize/inline landing. |
| **R3** | AllenaiLongformerBase | **−9.40%** (the #1-headroom model) | U11 `diagonal_skew_elimination` commits 12 read-bypasses but the 100MB band slice_scatter stays LIVE (read by softmax + epilogue) → never eliminated; bypasses just block fusion. | **VALIDATED FIX** — `R3_diagonal_skew_profitability_gate.patch`. Default-on gate `diagonal_skew_elimination_require_dead_scatter`: commit a bypass only if it makes the scatter FULLY dead. R3→659.4us (+149us recovered); also fixes an 11-repro b64f0e8a cluster (below baseline); pyhpc also improves. Bit-identical numerics. **Confidence HIGH on cause, MED-HIGH net-positive.** One tradeoff: `sum_444779f98932` 1006→1184us (gate too strict on a partial-bypass shape) — net strongly positive; a refined min-bytes gate would recover it. **Required before landing U11.** |
| **R2** | pyhpc_turbulent_kinetic_energy | **−12.38%** | **Diffuse** — one 1522-op pointwise stencil; many individually-correct scatter/slice passes each break a fusion (78→91 kernels, ~1M-elem each → extra HBM round-trips). | **NO single fix.** slice_scatter_elision-off recovers ~45% of gap; all-scatter-passes-off recovers 35/41us; no single lever. Real fix = a global "don't increase post-fusion kernel count on large pointwise graphs" guard. **Acceptable diffuse collateral** for now; flag as a follow-up scheduler guard, not a landing blocker. |

---

## 5. PR PLAN (ordered)

**Tier 0 — Headline, gated on verification**
1. **rsqrt_canonicalization (U20, `a73d1583b34`, +5.64pp)** — **OPEN FIRST, but only after** `rsqrt_verify/REPORT.md`
   confirms the fusion-unlock mechanism. Standalone post_grad PR with `config.rsqrt_canonicalization`. If verify
   refutes → pull from the plan and re-measure the branch total.

**Tier 1 — Clean, above-noise, independent LAND PRs (open in parallel)**
2. **CE online-softmax + loop-invariant hoist (U31, `ab29345d82d` +1.22, with mega-commit's CE prim hunks)** —
   selective-revert PR carrying `cross_entropy_loss_online` prim + ops_handler + lowering + triton hunks + the hoist.
3. **online-softmax fast combine (U28, `a26fc2c8bf4` +0.83)** — `config.triton.online_softmax_fast_combine`.
4. **scalar_acc family + gate (U29, `ca8f961d6b0` +0.59 and `9dde2c59a51` +0.44 bundled)** — scalar accumulators with
   the `num_load ≤ 4` gate and ungate-from-CD; ship the gate *with* the feature (it's a regression-fixer).

**Tier 2 — Independent fx-pass batch (small positives, own flags, bundle or small PRs)**
5. **select_scatter_sparsity + topo-fix (U06, incl. `f595e0cac3a` +0.50)** — pass + its ordering fix as one PR.
6. **rotate_half_gather (U15, +0.37)**, **structured_scatter_decode (U05, +0.13)**, **index_through_norm (U16)** —
   independent fx_passes, each own flag; batch as small PRs depending only on the post_grad registry.
7. **scatter_reduce_fusion family (U01/U03, incl. `4427bfd553c` +0.35 dtype-cast)** — depends on the mega-commit's
   `scatter_reduce_fusion.py` file existing; land after the enabling-file PR.
8. **split-reduction cost-model family (U22/U23/U24, segment-split +0.25, split-for-fusion)** — one PR (co-evolving
   in choices.py/ir.py/scheduler.py).

**Tier 3 — Requires regression fix bundled in**
9. **U25 realize/inline family (`905450a5a5d` + fix-chain) + R1 patch** — land the inline pass **only with**
   `R1_inline_profitability_gate_fix.patch` in the same PR.
10. **U11 diagonal_skew_elimination + R3 patch** — land **only with** `R3_diagonal_skew_profitability_gate.patch`
    (default-on dead-scatter gate). Recovers the Longformer regression.

**Deferred follow-ups**
- **Mega-commit decomposition (`97385fb3273`, +1.95pp, DECOMPOSE-LATER):** flag-ablation / selective-revert within
  the commit to attribute its ~9 units individually; then fold each into the Tier-1/2 PRs above. Lower priority than
  rsqrt verify. Drop reduction_chaining.py during this.
- **R2 pyhpc global guard:** scheduler-level "don't increase post-fusion kernel count on large pointwise graphs."

**Drops (do not open PRs):** BN-affine-fold (U17, net −0.76), reduction_chaining.py (dead), realize_reads_threshold
4→30 (moot, reverted), persistent-reduction inner-threshold 1024→8192 (moot, reverted), num_stages tune (net-zero
pair), degenerate-dropout-elim (net zero), realize_hint broadcast (net zero), hoist_invariant_compute (default off).

---

### One-line plan summary
Open **rsqrt first (pending kernel-verify)**, then the clean independents — **CE-hoisting, online-softmax-combine,
scalar_acc-gate, topo-fix/select_scatter_sparsity**, then the small fx-pass batch; land **inline (U25) only with the
R1 patch** and **diagonal_skew (U11) only with the R3 patch**; **defer** the mega-commit decomposition; **drop**
BN-affine-fold, reduction_chaining, and both reverted threshold changes.

## rsqrt #1 — VERDICT RESOLVED (verify done, results/pytorch_landing/rsqrt_verify/REPORT.md)
LAND ✓ (was VERIFY-FIRST). a73d1583b34 rsqrt-canonicalization is REAL, reproducible (repvgg
compiled 2.46x / cd 2.97x; mobilenet 2.00x/2.12x), numerically sub-bf16-ULP, clean adjacent-edge
attribution. MECHANISM (corrected — both fusion-unlock AND autotune theories refuted): swaps a
~30-instruction correctly-rounded SOFTWARE sqrt_rn+div, run per-element over 102M elems (BN scale
not hoisted), for ONE rsqrt.approx.ftz hardware MUFU instr -> kernel goes compute-bound(2.5-3.7x SOL)
-> memory-bound(~1x). Micro-bench confirms (single-branch 1.23x, two-branch BN 2.27x).
**MAGNITUDE HONESTY: the +5.64pp / 2.2x is PER-KERNEL/geomean-over-kernel-ratios. Per-MODEL
wall-clock is ~1.15x on affected conv models (repvgg 5005->4358us). Quote per-kernel, NOT per-model.**
This means the per-commit-geomean attribution (which is kernel-ratio-based) OVERSTATES model impact
for kernel-localized wins like rsqrt — a caveat for the whole A1 table: kernel-geomean != model-e2e.
PR: clean 45-line peephole, lands first. Footnote rsqrt.approx is ftz + marginally less precise (accepted, sub-bf16-ULP).

## UNITS CORRECTION + refinement-validation (2026-06-23) — CANONICAL = per-model-e2e
The A1 walk attributed in KERNEL-geomean (fusible-only ratio); the honest model number adds the
extern denominator (rollup.py formula). Recomputed from existing state data (no re-bench):
FULL_per_commit_e2e.txt. Magnitudes correct DOWN ~2-3x (uniform extern dilution):
  rsqrt +5.66->+2.03pp e2e | mega +1.93->+0.72 | CE-hoist +1.22->+0.54 | online-softmax +0.83->+0.33
  | thresh +0.41/-0.37 (cancel) | BN-affine -0.29 (still net-neg) | online-softmax-seg +0.57
REFINEMENT VALIDATION: checked every coarse gap kernel-geomean vs e2e — ZERO decision flips. Every
mover stays a mover, every flat gap stays flat in e2e (dilution is ~uniform, not selective). So the
gaps we drilled were the RIGHT ones; only magnitudes change, NOT the ranking or which-to-pursue.
PR PLAN RANKING UNCHANGED (rsqrt #1, mega, CE-hoist, online-softmax, scalar_acc...); cite e2e pp.

## MICROBENCHMARK STABILITY — RESOLVED 2026-06-23 (the ±10-44% swing was NOT real)
**The "genai microbenches are noisy" caveat below is WITHDRAWN.** A dedicated stability investigation
(results/pytorch_landing/genai_stability/, agent aca9f81fd4) re-benched the 3 representative micros 8x
cold + 8x warm at fixed HEAD through the PROPER harness (bench_parallel.py --full-graphs, do_bench
min-of-100, per-GPU lock). Run-to-run spread = **0.03-0.42%** — three orders of magnitude tighter than
the ±25% I'd seen. The decisive test: across 8 cold recompiles the Triton SOURCE is bit-identical and
autotune picks the IDENTICAL config every run (SoftmaxForward 16384/w8, CE 8192/w4). So NONE of the
candidate mechanisms is live: not autotune config-selection nondeterminism, not libdevice->PTX
bimodality (exp is pinned tl_math, zero libdevice at HEAD), not clock/runtime variance. Clock-locking
is UNAVAILABLE (perms) AND UNNECESSARY (clocks steady-at-boost 1852MHz, no throttle).
ROOT CAUSE of the apparent swing: the original genai per-commit series was computed on a DIFFERENT,
noisier path — almost certainly a bespoke/concurrent timing that skipped the GPU lock (the classic
"never new benches, reuse" failure mode). Re-benching base 5e2ab + both "Update" merges through the
locked path gives a stable 3462us (spread 0.006%): **the merges do NOT move it; the ±10-44% attributed
to them is SPURIOUS and is discarded entirely.**
The REAL genai move is ONE step at `9dde2c59a51` (scalar-acc reduction config, "ungate scalar-acc
from CD + raise rnumel ceiling"): SoftmaxForward 3541->1905us, a genuine ~46% win, stable on both
sides. exp-path (fast-math) is secondary: forcing use_fast_math at base gives only ~14%; the dominant
~36% is the reduction structure, NOT exp. The pre-flip commit a85d79a900a already emits tl_math.exp
yet is 3539us -> exp is not the lever.
**GENAI DETECTION THRESHOLD: ±0.5%** (conservative; measured floor 0.03-0.38%). Tighter than the
model-level A/A band (±0.82%) because each genai micro is one large bandwidth-bound kernel timed
min-of-100 — no per-kernel jitter to average. Any genai per-commit delta >~1% is real signal.
STABILIZATION = procedural (no flags, no extra reps, no clock lock): recompute genai per-commit
deltas through bench_parallel.py --full-graphs and the swing is gone.

### GENAI MOVER A/B — TARGETED, LOCKED PATH (2026-06-23, replaces the spurious series)
Re-benched genai at the 4 mover A/B pairs ONLY (not all 71 commits) via the locked --full-graphs path
(results/pytorch_landing/genai_stability/mover_ab/, agent a43768c). Worst spread 0.52%, mostly <0.1% —
STABLE, confirming the harness was never noisy. The intermediate states CHAIN coherently (SoftmaxForward us):
BASE 3462 -> [mega] 3929 -> [online-softmax] 3539 -> [scalar-acc] 1905 = HEAD (each mover's predecessor
matches the prior mover's output to 3 sig figs). CORRECTED per-family attribution:
- **scalar-acc `9dde2c59a51`** = the BIG genai lever: Softmax +46%, CE +64% (fwd&bwd), config flip. (was U29)
- **online-softmax `a26fc2c8bf4`**: Softmax +10%, CE +20% (fwd&bwd). CE-fwd is same-config-faster; rest config flips. (U28)
- **rotate_half `1406552b9d3`**: moves NONE of the 8 genai micros (all <=0.26%) — its RoPE/gather win isn't in this micro set. (U15)
- **mega-commit `97385fb3273`**: touches all 4 families but is NET-NEGATIVE on micros IN ISOLATION (Softmax/CE -13..-25%).
  The squashed commit is an intermediate state; the family WINS land in the later dedicated movers above, NOT in the mega.
  => Refines the prior "mega carries the CE/softmax transforms" claim: mega changes the codegen PATH, the speedups are
  attributable to scalar-acc + online-softmax. Measure the mega DECOMPOSITION against genai, but do NOT credit it the wins.
- **RESOLVED — the norm-backward 2x does NOT ship (HEALED at HEAD).** Net HEAD-vs-BASE check (all 8 micros fwd+bwd,
  locked path, worst spread 0.53%): RMSNormBackward/LayerNormBackward = BASE 395/392us -> mega 817/817us (the 2x) ->
  a85d79a900a 307us -> HEAD 306.9/307.0us. The 817us is an artifact of benching the SQUASHED mega-commit in isolation
  (a synthetic aggregate state that never appears as a real branch HEAD). Mechanism (kernel-diff): the regressed kernel
  is the dw weight-gradient reduction (mul_sum_1) — mega emits it LOOPED (@reduction, multi-pass), BASE+HEAD emit it
  single-pass (@persistent_reduction). The MixOrderReduction fixed_config split kernel is bit-IDENTICAL mega vs HEAD
  (NOT the cause). HEAD actually beats BASE on norm-backward (+22%) by making ALL these reductions persistent
  (RMSNormBackward kernel-types: BASE 4persist+2loop, mega 4persist+2loop+2fixed, HEAD 8persist+2fixed, ZERO looped).
  => 846f6be7015 / U30 MOR path = NO shipped regression. NET genai HEAD-vs-BASE: fwd-only +53.0% geomean (1.530x),
  incl-bwd +46.8% (1.468x); every micro improves or is at-floor, no net regression anywhere. (mover_ab/head_vs_base/)

### (superseded) earlier NOISY reading — kept for provenance
Computed per-commit deltas on the 8 genai micro-benchmarks (CE/softmax/layernorm/rmsnorm fwd+bwd).
CAVEAT: micro-benches are single tiny kernels with NO averaging -> per-commit deltas are VERY NOISY
(SoftmaxForward swings +/-10-44% on pure "Update" MERGE commits that change nothing). Do NOT cite
individual micro per-commit deltas as attribution. They corroborate at the FAMILY level only:
- mega-commit 97385fb3273: CrossEntropyForward +121%, CEBackward +35%, SoftmaxBackward +44% -> the
  mega-commit genuinely transforms CE/softmax (it contains CE-gather-into-softmax + MOR). Confirms
  the genai wins live in the mega-commit -> measure its decomposition AGAINST genai.
- a26fc2c8bf4 online-softmax-fast-combine: CEForward +35%, CEBackward +11% -> real, on-target.
- 1406552b9d3 rotate_half_gather (RoPE): SoftmaxForward +61% -> plausible.
USE: microbenchmarks = a SEPARATE caveated "raw kernel win on targeted ops" view (high variance),
NEVER folded into the model headline (policy). Useful to confirm WHICH op-family a commit targets,
not for precise pp. For the mega-commit decomposition, genai is the right measurement surface for
its CE/softmax/MOR features.

## §6 — MEGA-COMMIT FLAG-ABLATION RESULTS (2026-06-23, RESOLVES the DECOMPOSE-LATER line)
Ablated AT HEAD (HEAD-default control vs HEAD-with-one-flag-toggled = MARGINAL contribution), locked path only
(bench_parallel repro path + --full-graphs genai), per-pass counters to find affected repros. Full data:
results/pytorch_landing/mega_ablation/RESULTS.json.

**EMPIRICAL kernel-geomean noise floor (answers the open kernel-floor question):** two independent HEAD-default
control runs → kernel-geomean control-vs-control = **1.0017 (~0.2%)**, per-shape up to ±9% on sub-10us kernels.
So the kernel-geomean floor is ~0.2% — MUCH tighter than the ±0.82pp MODEL-e2e floor (geomean averages ~1727
ratios → per-point noise shrinks ~sqrt(N)). CONFIRMS: a +0.35pp kernel-geomean span is REAL signal, not noise;
the structural tell of a genuine codegen change is an n_kernels change (not the source hash — inductor embeds a
config-hash COMMENT so the hash differs even when a pass didn't fire; counters['inductor'][pass] is authoritative).

| Unit | Flag | Affected repros | Kernel-geomean (off/ctrl) | Verdict |
|------|------|----------------|---------------------------|---------|
| U01 scatter_reduce_fusion | env (default OFF) | **0/1727** | n/a (zero fires even forced ON) | **DROP-on-corpus** — pattern absent from corpus; self-report only |
| U07 slice_scatter_elision | env | 18 (Longformer, pyhpc) | 1.0067 (at floor) | **BUNDLE** — 2 real pyhpc structural wins, Longformer mass flat |
| U08 as_strided_scatter_elision | env | 10 (CNN-BN) | 1.0043 (at floor) | **BUNDLE/borderline** — carries a tf_efficientnet 0.75x regression; gate it |
| U09 linear_reduction_elimination | env | 6 (functorch_dp_cifar10) | **0.9929 NET-NEG** | **DROP-net-negative** — flag-off is faster; nk never changes; needs cost-model gate |
| U10 layout_transform_store_sinking | env | 14 (shufflenet_v2) | **1.2003 REAL** | **LAND** — structural (nk 1→3 fuses channel-shuffle), far above floor; verify per-model on ShuffleNet roll-up |
| U25 inline_recomputable_producers | config-patch (NO env) | 746 (~2/66 structural) | **0.9826 NET-NEG** | **DROP/decompose** — Longformer 1.36x win buried under XLNet 0.62x / XGLM 0.66x / GPTNeo 0.77x reproducible regressions; = the pytorch_unet -0.50% source; needs cost-model gate |
| U30 MOR finalize | NONE (unconditional) | — | — | **SELECTIVE-REVERT** — not env-ablatable; doesn't fire on genai; right surface = Swin roll-up |
| U31 CE gather-into-softmax | NONE (prim) | — | — | **SELECTIVE-REVERT** — right surface = CrossEntropy* micros (locked --full-graphs); revert prim+ops_handler+lowering+triton CE hunks together |
| U33 Blackwell-BN + low-warp | PARTIAL (1 env scalar) | — | — | **SELECTIVE-REVERT** — autotune config-list code changes; the d2cd/62f27 inner-threshold was add+revert net-zero (DROP-MOOT) |

**Genai (locked --full-graphs, ISOLATED):** none of the 6 ablatable flags fire on any genai micro → zero movement.
METHODOLOGY CATCH (re-confirms the lock lesson): the agent's FIRST genai off-arms ran 3-concurrent on separate GPUs
and showed bogus 3-19% "improvements" with 6-8% spread — the SAME spurious-swing artifact. Re-run in ISOLATION:
zero movement, <0.5% spread. The genai-relevant features are U30/U31 (no-flag), not the 6 ablatable passes.

**RECONCILIATION:** the walk's mega +1.95pp kernel-geomean / +0.71pp e2e is NOT evenly spread. On the measured
corpus+genai surface only U10 is a clean positive; U09+U25 are net-negative. The bulk of +1.95pp is therefore in
the 3 NO-FLAG units (U30 Swin-MOR 35%-claim, U31 CE, U33 BN/low-warp) — whose surfaces (Swin roll-up, CE pattern)
env-ablation can't reach. REVISED PR GUIDANCE: land U10 clean; bundle U07/U08 below floor (gate U08's efficientnet
regression); DROP U01 (corpus-dead); DROP-or-gate U09 + U25 (net-negative, U25 most urgent — it's the pytorch_unet
regression); the 3 no-flag units need selective-revert arms (U31 against CE micros, U30 against Swin) to attribute.

## §7 — MEGA-CORE SELECTIVE-REVERT (U30/U31/U33 RESOLVED, 2026-06-23) — overturns the §6 assumption
The 3 no-flag mega units (which §6 GUESSED carry "the bulk" of the +1.95pp) are now MEASURED by selective-revert
at HEAD (revert just that unit's hunks, re-bench affected surface, locked path + clocks pinned 1852). Every revert
verified to change codegen before timing. Data: results/pytorch_landing/mega_ablation/CORE_REVERT_RESULTS.json.
Kernel-geomean noise floor measured here = ±0.2% (control-vs-control 1.0014).

| Unit | What | Surface | kernel-geomean (ON/OFF) | Verdict |
|------|------|---------|------------------------:|---------|
| **U30** MOR Triton finalize-sum | simd.py `_emit_triton_finalize_sum` | 52 sum_sum_sum_* (BN-bwd/sibling-reduction tails) | **0.881 = 1.135x faster** (CD 1.20x), 48/52 help, 0 regress | **LAND-worthy** — real no-flag win. Confirms "35% Swin" DIRECTION (geomean ~13.5%; 35% = best-case shape). |
| **U31** CE gather-into-softmax | inductor_prims/ops_handler/lowering/triton CE hunks | 14 amax_sum_sum_* (Electra/GPT2/GPTNeo/Roberta) — NOT the genai CE micros (pattern breaks on bf16 cast) | **1.027 = ~2.6% SLOWER** (CD 1.014), ALL 14 regress | **DROP / net-negative** — REFUTES the "0.8%" self-claim. Matches the commit's own "limited by loop-invariant hoisting" hedge. |
| **U33** Blackwell-BN/low-warp/coop-widen | triton_heuristics.py | var_mean BN repros | **0.740 = 1.35x faster** on var_mean_mean (up to 1.81x ghostnet); plain var_mean at floor | **Real kernel win but per-model-e2e ~0** AND **NOT in the mega-commit** — these are 3 SEPARATE follow-up commits (7be29e49d9e, 2fbbe401871, 2b35f4ee83a), not mega hunks. Sub-100us BN kernels invisible in multi-ms CNN e2e. |

**RECONCILIATION — the §6 premise is FALSIFIED.** The mega's +1.95pp kernel / +0.71pp e2e is NOT carried by the
no-flag units as a group. Only **U30** is a positive mega contributor; **U31 is net-NEGATIVE** (subtracts from the
mega's geomean); **U33 isn't in the mega-commit at all** (scope error in the original A0 mapping). So the mega gain =
**U10 (flag win) + U30 (no-flag win)** partially offset by net-negatives **U25 + U09 + U31**. NO single hidden unit
accounts for the gain — it's the residual of several modest localized wins minus several modest localized losses,
diluted ~2-3x to the honest +0.71pp e2e.

**REVISED MEGA PR GUIDANCE:** land **U10** + **U30** (the two real wins; U30 is a clean no-flag MOR-finalize PR, bench
against Swin/sum_sum_sum); DROP **U01** (corpus-dead), **U09** (net-neg), **U31** (net-neg, refutes its own claim);
gate-or-drop **U25** (net-neg, the pytorch_unet regression). U33's BN/low-warp commits are SEPARATE follow-ups —
real kernel wins, ~0 model-e2e, land on their own merits if at all. This closes the mega decomposition: every unit
is now individually measured, not assumed.
