# PHASE 1 — PyTorch-main Landing: Commit Inventory & Phase-2 Benching Plan

**Date:** 2026-06-19  ·  **Mode:** read-only archaeology (CPU only)  ·  **Status:** analysis complete, no history rewritten, nothing benched, nothing pushed.

- Repo / branch: `tmp_work` in `/tmp/pytorch-work`, HEAD `daa79cd25ca` (owner WIP present in tree: `config.py`, `scatter_reduce_fusion.py` — untouched).
- A/B baseline given in task: `244fdb379d11`.
- Companion machine-readable file: `PHASE1_commit_inventory.json` (this directory).

---

## 0. TL;DR (the numbers that matter)

| Quantity | Count |
|---|---|
| `git rev-list --count 244fdb..HEAD` (raw) | **139** |
| …non-merge | 137 |
| …merges | 2 |
| **Upstream PyTorch-main PRs the branch was rebased onto** (`244fdb..5e2ab`) | **56** ← *NOT perf-branch work* |
| **Perf-branch commits** (`5e2ab..HEAD`, non-merge) | **81** |
| …of those: ghstack `Update` plumbing | 5 (+2 merges) |
| …revert commits | 6 |
| …reverted commits (net-zero with their revert) | 6 |
| …superseded (hunks overwritten by a later commit) | 3 |
| **SUBSTANTIVE SURVIVING commits** | **61** (+1 benchmarking-infra change hidden in the plumbing) |
| **Feature units the 61 collapse into** | **~24** (11 cleanly separable + 1 entangled core) |

**Headline for the user:** "139 commits" is a mirage. **~56 are upstream PRs that came in via two ghstack `base update` merges** and are already landing on main independently — they are *not* this branch's contribution. The branch's own work is **81 commits, of which 61 are substantive and survive in HEAD**, and those 61 collapse to **~24 feature units**. The truly *independent, individually-landable* surface is **~11 new FX-pass files**; everything else is one **entangled codegen/scheduler core** anchored by a single 5,237-line squashed "mega-commit."

---

## 1. Branch topology (the critical structural finding)

The history is **linear** (ghstack first-parent chain), not a tangle of branches:

```
244fdb379d11  (A/B baseline)
   │  56 upstream PyTorch-main PRs  (#NNNNNN-tagged: _scaled_mm_v2, atomic-add epilogue,
   │                                 dynamic-slice lowering, TF32 pad_mm, flex vectorization, …)
   ▼
5e2ab3055de  ← UPSTREAM TIP the branch sits on  (== merge-base(HEAD, 5e2ab); verified ancestor of HEAD)
   │  ┌── ac93f5b2845  "Update (base update)"  ── merges upstream 5e2ab in
   │  ├── aa798f9f052  "Update"
   │  ├── 47aafe246e1 / 9c296b0e9b4 / a1c8cd8e1c9 / 111efb8724a  "Update"  (benchmarking.py ghstack churn)
   │  ▼
   │  97385fb3273  "[inductor] All kernel optimization improvements (June 2026 session)"
   │               ← MEGA-COMMIT: 5,237 insertions / 17 inductor files / 8 brand-new pass files
   │  ▼
   │  ~50 incremental [inductor] commits (extensions + fixes + 6 revert-pairs)
   ▼
daa79cd25ca  (HEAD)
```

### Consequence — the baseline is wrong for attribution
`git diff 244fdb..HEAD -- torch/_inductor` contains **both** the perf work **and** ~9 upstream inductor PRs (they show up only because `244fdb` predates them). The clean A/B for *this branch's contribution* is:

> **`5e2ab3055de .. daa79cd25ca`**, not `244fdb379d11 .. HEAD`.

Upstream inductor PRs that pollute the `244fdb` diff (exclude from the landing set; they land separately):
`_scaled_mm_v2 (#182527)` → `kernel/mm.py`+210; `atomic-add epilogue fusion (#184179)` → `scheduler.py`; `Fix dynamic slice lowering (#183652)` (and its revert #183652) → `lowering.py`/`ir.py`; `skip TF32 padding non-contig fp32 mm (#184029)` → `pad_mm.py`+39; `Generalize flex flash vectorization (#185020)` → `kernel/flex/*` (+~640); `Preserve narrow dtype convert chains (#184366)` → `joint_graph.py`; `mix-order reduction data-dependent guards (#184870)` → `scheduler.py`; `cat lowering 1-D empty (#185549)` → `lowering.py`.

**Phase-2 action item:** bench A=`5e2ab3055de`, B=`daa79cd25ca`. (If you must use `244fdb` as A, the +4.84% number is *contaminated* by the upstream inductor PRs above — re-confirm against `5e2ab`.)

---

## 2. Classification of the 81 perf commits (+2 merges = 83 in `5e2ab..HEAD`)

| Class | N | Meaning |
|---|---|---|
| **SURVIVE** | 61 | a hunk it introduced is still present in HEAD's product files |
| **REVERT** | 6 | a `Revert "…"` commit |
| **REVERTED** | 6 | undone by its later revert (net-zero pair) |
| **SUPERSEDED** | 3 | touched product but its specific hunks were overwritten by a later commit |
| **PLUMBING** | 7 | 2 base-update merges + 5 `Update` ghstack-iteration commits |

Reconciles: 61+6+6+3+7 = **83**. ✔

### 2a. Revert-pairs (explicit map) — 6 pairs, all net-zero
| Reverted commit | Reverted by | Subject |
|---|---|---|
| `1a824e0747a` | `f74a7f13723` | Raise realize_reads_threshold 4→30 *(see ⚠ below)* |
| `9ae68049a93` | `282760481f8` | Smart realize_hint: don't materialize broadcast-dominated producers |
| `caa0b614314` | `f0b54e1aa1f` | Tune num_stages for memory-bound pointwise kernels |
| `d2cd52d3117` | `62f27911a59` | Raise persistent-reduction threshold for INNER reductions 1024→8192 |
| `84a2a5cf6ae` | `f654a5d3a0b` | Degenerate dropout elimination pass (var_mean_81e1858d9aa4) |
| `13cbc75b516` | `1787656f21b` | Degenerate dropout elimination pass (var_mean_7b0bd4f35599) |

> ⚠ **Mislabeled revert — `f74a7f13723`.** Its subject says *Revert "…realize_reads_threshold…"*, and it does revert the `config.py` knob (`config.py | 2 +-`). But it **also adds 28 lines to `torch/_inductor/memory.py` that survive in HEAD**. So this commit is *partly substantive*. It's the sole source of the surviving `memory.py +28` and is the only "unexplained" product file the revert-graph couldn't attribute. For Phase 2, treat `memory.py` as part of the **scheduler/realization** unit, not as net-zero.

> Note: "Degenerate dropout elimination" appears **twice** (two independent add→revert cycles, different repros) — the pass was tried, reverted, retried, reverted again. Net contribution = **0**. Do not land.

### 2b. Superseded (3) — fold into the unit that rewrote the lines
- `f9493caabcd` "Relax pointwise_cat guard" — 4 deletions in `lowering.py`, later region rewritten.
- `be6874e5fa2` "Extend inline_recomputable_producers (multi-input stencil)" — 5/-4 in `scheduler.py`, overwritten by later scheduler fixes.
- `8586e404cc8` "Tighten aggressive split to 2/3 SM" — 1 line in `choices.py`, overwritten by a later split-threshold commit.
These intents may live on inside their unit, but as standalone commits their hunks are gone → not individually landable.

### 2c. Plumbing (7)
The 5 `Update` commits churn `benchmarking.py` + `test_benchmarking.py` enormously across ghstack iterations but **net to a stable `benchmarking.py +47/-1` and `test_benchmarking.py +148`** — identical at the pre-mega-commit point and at HEAD. So there *is* one surviving change here, but it's **autotune-timing infra**, not kernel codegen → expected ≈ **zero model-perf impact**. Collapse the 5 commits to **one logical "runtime/benchmarking" change** and bench it only if you care to confirm 0%.

---

## 3. Reconstruction cross-check (does the surviving set rebuild the product diff?)

Product diff `5e2ab..HEAD -- torch/_inductor + value_ranges.py` = **36 files, +13,821 / −200**.

- Files touched by SURVIVE commits: **34 / 36**.
- 2 "unexplained" files, both resolved:
  - `runtime/benchmarking.py` → carried by the 5 `Update` plumbing commits (§2c).
  - `memory.py` → the mislabeled revert `f74a7f13723` (§2a ⚠).
- **0 "extra" files** (no surviving commit claims a file absent from the net diff).

> **Verdict: reconstruction is complete.** The 61 SURVIVE commits + 1 plumbing benchmarking change + the memory.py addition fully account for the product diff. No residual the attribution can't explain.

---

## 4. The SURVIVING logical-commit set (ordered, oldest→newest, with subsystem + unit)

61 commits. Subsystem tags: scheduler / codegen-triton / codegen-simd / triton_heuristics / fx_passes / config / lowering / ir / sympy-value_ranges. Full per-commit table (all 83, incl. class) is in **`PHASE1_commit_inventory.json` → `ordered_commits`** and reproduced in `_table.md`. Grouped by **feature unit**:

### Entangled CORE (NOT independently landable — share scheduler/codegen/cost-model surfaces)

**Anchor — the mega-commit (must land first; everything below depends on it):**
- `97385fb3273` "All kernel optimization improvements (June 2026 session)" — *seeds* 7 new pass files (`scatter_reduce_fusion`, `scatter_reduce_gather`, `slice_scatter_elision`, `as_strided_scatter_elision`, `linear_reduction_elimination`, `layout_transform_store_sinking`, `reduction_chaining`) **and** the `inline_recomputable_producers` scheduler machinery, evict_first/scalar_acc triton changes, `post_grad.py` registration, `ir.py`/`lowering.py`/`config.py` hooks. 5,237 insertions. *This is ~⅓ of the entire surviving diff in one commit.*

**Unit: scheduler heuristics — `inline_recomputable_producers` / realize / mutation_renames / MOR fusion** (10 commits) — *R1 pytorch_unet lineage*
`14b0254f8a9, 905450a5a5d, 846f6be7015, 12e839eb0fc, 7be29e49d9e, c465f1751c6, ec674714f38, 054c68b5e15, a6e9664fb72` (+seed in mega-commit; +superseded `be6874e5fa2`).

**Unit: triton codegen — evict_first / scalar_acc / online-softmax combine / CE-hoisting** (7) — `codegen/triton.py`,`simd.py`
`ca8f961d6b0, 52d4cadfac0, f6ae31e6984, e45f846f8d1, 60431eca431, e9a67c98a8c, a26fc2c8bf4`.

**Unit: split-reduction cost model** (7) — `choices.py`,`config.py`
`2b35f4ee83a, 2fbbe401871, d75864dea06, 5d4ce5df93b, ab46039c492, bdc289b3644, a85d79a900a` (+superseded `8586e404cc8`).

**Unit: triton_heuristics — persistent-reduction configs / low-warp / rnumel ceiling** (2) — `runtime/triton_heuristics.py`
`44802547aa9, 9dde2c59a51`.

**Unit: ir — materialize-on-reuse / indirect-index / multilayer split** (3) — `ir.py`
`b1e8a30af15, 715c6fc42ac, d85ab5508e3`.

**Unit: lowering — clamp-embedding / pointwise_cat / CE-hoist** (3) — `lowering.py` (+`value_ranges.py` for the clamp)
`ab29345d82d, aa83bc1951c, d95f59fb1bc` (+superseded `f9493caabcd`).

**Unit: runtime/benchmarking infra** (plumbing, 1 logical) — autotune timing; ≈0% expected. The 5 `Update` commits.

**Unit: memory (realize_reads sizing)** — `memory.py` +28 via the mislabeled revert `f74a7f13723`.

### Independent FEATURE UNITS (each a self-contained new FX-pass file → individually landable)
*Created by a single later standalone commit (file did not exist before it). Only coupling is a few lines in the shared `post_grad.py` registry + occasional small touch of mega-commit's `scatter_reduce_fusion.py`.*

| Unit (new file) | Commit(s) | Notes |
|---|---|---|
| `structured_scatter_decode` (affine-iota index_put overlap-add) | `56959375c33` | standalone |
| `select_scatter_sparsity` | `94ad050dee0` (+fix `f595e0cac3a`) | standalone |
| `slice_output_compaction` | `bbf37d454c2` (+`3b746fce660`,`3bf69043be0`) | standalone |
| `diagonal_skew_elimination` (Longformer scatter-read bypass) | `d8e9914094a` | ⚠ also edits `scatter_reduce_fusion.py` +13/−2 → small dep on mega-commit. **R3 Longformer.** |
| `one_hot_reduction_elimination` | `049d1229d84` | standalone |
| `expand_sum_elision` | `c33b0e78618` | standalone |
| `cat_through_reduction` | `6fdaadaa2c0` | standalone |
| `rotate_half_gather` (RoPE) | `1406552b9d3` | standalone |
| `index_through_norm` | `afc3d5956d5` | bundled w/ "elide constant-index bounds checks + split-for-fusion" |
| `bn_affine_folding` | `86c5451e197` | standalone |
| `dedupe_graph_outputs` | `30cc64e2343` (+`60dd3839913`,`daa79cd25ca`) | standalone; 3 commits = pass + 2 canonicalization/recursion fixes |
| `expand/constant-fold uniform thru view` | `3e627768c9f`,`565695da70b` | `joint_graph.py` ConstantFolder |
| `scatter_reduce_fusion` *extensions* | `e034b931eab, 290b0f22d9a, 4427bfd553c, e6d89ca0b31, edbd4b67279, 5489b8c2bb9` | **depend on mega-commit** (extend its file) — NOT independent |

> Caveat on auto-grouping: the JSON's `unit` field assigns each commit to the *first new-pass file it touches*; a handful land oddly (e.g. `a73d1583b34` "rsqrt canonicalize" shows as "post_grad glue"; `9b4edfffc15` "CE hoist" shows under slice_scatter_elision because it also touches that file). The groupings in this section are the hand-curated truth; treat the JSON `unit` as a starting index, not gospel.

---

## 5. Entanglement verdict

**Mixed — and the split is the whole story for landability.**

- **NOT a clean set of 30+ independent commits.** ~⅓ of the surviving diff is one squashed mega-commit (`97385fb3273`), and ~30 of the 61 survivors are **extensions/fixes to surfaces the mega-commit created** (scheduler `inline_recomputable_producers`, triton evict_first/scalar_acc, choices split cost-model, `scatter_reduce_fusion.py`). These cannot be cherry-picked apart without the mega-commit and largely without each other (later fixes patch earlier hunks in the same functions).
- **~11 new FX-pass files ARE genuine independent feature-units** (each is a new file, registered via a few lines in `post_grad.py`). They can in principle be lifted into standalone PRs. **Caveat:** the shared `post_grad.py` registry is touched by **20 commits** — a *textual* coupling (merge-order/registration) but not a *functional* one; each pass's logic is self-contained. `diagonal_skew_elimination` is the one "independent" pass with a real (small) code dependency on the mega-commit's `scatter_reduce_fusion.py`.

**Feature-unit seams (natural PR boundaries):**
1. **CORE bundle** (un-splittable): mega-commit + scheduler-realization unit + triton-codegen unit + split-reduction cost-model unit + triton_heuristics + ir + lowering + memory + benchmarking. *Lands as one large PR (or a tightly-ordered ghstack), not as independent pieces.*
2. **Independent pass PRs** (each optional, gated by its own config flag): structured_scatter_decode, select_scatter_sparsity, slice_output_compaction, one_hot_reduction_elimination, expand_sum_elision, cat_through_reduction, rotate_half_gather, index_through_norm, bn_affine_folding, dedupe_graph_outputs, constant-fold-uniform. (`diagonal_skew_elimination` ≈ independent but needs a 13-line scatter_reduce_fusion shim → land after CORE.)

**Phase-2 implication:** per-*unit* attribution is feasible for the 11 independent passes (toggle each pass's config flag A/B). For the entangled CORE, attribution must be **cumulative** (walk the commit chain) — you cannot isolate "the evict_first change" from "the split cost-model change" by file, because they co-evolve in `triton.py`/`choices.py`/`scheduler.py`.

---

## 6. Phase-2 benching plan + cost estimate

**Given:** a full all-shapes A/B arm ≈ **36 min on 4×B200** (one arm). Call one A/B sweep ≈ 2 arms ≈ **~72 min ≈ 1.2 GPU-arm-hours of wall** (4×B200). Below, "1 sweep" = one all-shapes run of one state.

### Option A — cumulative checkout-walk on the original branch
Bench at each *surviving-state* boundary; impact_K = state_K − state_{K−1}. Reverts auto-cancel in impact-space.
- Naive (all 83 states): 83 sweeps × 36 min ≈ **50 wall-hours** on 4×B200. **Too expensive.**
- Pruned (skip pure-test/plumbing/reverted; bench only the 61 survive-boundaries + endpoints): ~63 sweeps ≈ **38 wall-hours**. Still a big spend.
- **Recommended pruning:** bench at **unit boundaries**, not per-commit. ~24 unit-endpoints + baseline `5e2ab` + HEAD ≈ **26 sweeps ≈ 15.6 wall-hours** (≈ 62 GPU-hours across 4×B200). Captures per-unit impact; within-unit fixes are summed (which is what you want — a fix is part of its feature).

> ⚠ Walk hazard (from MEMORY.md *CUDAGraph launch-floor* + *compile_fx not kernel-faithful*): a checkout-walk **changes the working tree** — forbidden in the shared `/tmp/pytorch-work`. Do the walk in a **throwaway worktree** (`git worktree add … <sha>` per state, or one worktree you `git -C … checkout` inside). And reset dynamo per shape or dynamic-kernel contamination poisons adjacent states.

### Option B — cherry-pick a clean series onto baseline, bench each
Cleaner conceptually (drops reverts/churn), but the mega-commit + 30 entangled fixes will **conflict heavily** when reordered, and probing conflict-cleanliness for 61 commits is itself a chunk of work. **Probe required** (in `/tmp/scratch_space/pl-probe` throwaway worktree, removed after). For the 11 independent passes cherry-pick is clean; for CORE it is not. **Not recommended for CORE.**

### Option C — cheaper proxy first (RECOMMENDED to run before any GPU spend)
The 11 independent passes are **config-gated**. So the cheapest high-signal experiment is a **single A/B pair per pass via its config flag**, on B=HEAD with each pass toggled off — but that still needs GPU. The *truly* cheap CPU-only proxy: **per-commit "did codegen change?" diff** — for each adjacent surviving state, compile the affected repro set and diff the generated Triton; only states that change emitted kernels can move perf. This narrows the 24 units to the handful that actually alter codegen on the model set, before spending any all-shapes GPU time. Feasible on CPU (compile-only, no timing). **Do this first to rank units, then bench only the top movers.**

### Recommended concrete plan
1. **(CPU, ~free)** Re-confirm the +4.84% against the **correct** baseline `5e2ab3055de` (not `244fdb`) — the given number may include upstream inductor PRs. *This gate alone may change scope.*
2. **(CPU)** Codegen-diff proxy (Option C) over the model set per unit → rank units by "changes ≥1 model's kernels."
3. **(GPU)** Bench **only** the units that (a) change codegen AND (b) are plausibly above the **~0.82% A/A noise floor**. Expectation: the +4.84% is carried by a **few** units (the scheduler-realization R1 family + a couple of high-multiplier passes), not spread across 24. Budget **~6–10 targeted A/B sweeps ≈ 4–6 wall-hours (≈ 16–24 GPU-hours)** instead of the full 15.6h walk.
4. The **benchmarking-infra** and **memory.py** changes: assume 0% unless a unit-diff says otherwise — don't spend a sweep on them.

**Estimate summary:** full per-unit walk = **~15.6 wall-hours / ~62 GPU-hours** (4×B200). Targeted (proxy-first) = **~4–6 wall-hours / ~16–24 GPU-hours**. Per-commit walk = ~38–50 wall-hours (**don't**).

---

## 7. Known effects → surviving commits

Branch nets **+2.18% median / +4.84% geomean** (given); A/A floor **~0.82%**; 3 model regressions.

| Known effect | Surviving unit | Commits |
|---|---|---|
| **R1 pytorch_unet −20% [fix validated]** | scheduler-realization / `inline_recomputable_producers` + materialize-on-reuse | seed in `97385fb3273`; `905450a5a5d`, `12e839eb0fc`, `ec674714f38`, `2b40e059b84`, `a6e9664fb72`, `846f6be7015`, `715c6fc42ac`, `b1e8a30af15`; plus the reverted `9ae68049a93`/`282760481f8` "smart realize_hint" experiment. *This is the realization-policy family — matches the "fix realization in the scheduler" theme. The −20% and its fix both live in CORE → can only be measured cumulatively.* |
| **R3 Longformer −9.4% [fix validated]** | `diagonal_skew_elimination` | `d8e9914094a` (the only commit creating that pass; the "scatter read bypass for Longformer, 8→5 kernels"). *Near-independent unit; benchable in isolation via its config flag.* |
| **pyhpc −12% [diffuse]** | **unattributable by subject** | no commit mentions pyhpc; consistent with "diffuse." Will only surface in a cumulative walk or a codegen-diff over pyhpc specifically. |
| **+4.84% geomean (where it comes from)** | almost certainly concentrated in: the **mega-commit + scheduler-realization unit** (broad), plus high-multiplier passes (`scatter_reduce_fusion` family showed up to 3.60×→0.45x on repros; `dedupe_graph_outputs` 5.13×→0.71x; `structured_scatter_decode` 2.85×). | The per-repro ratios in commit subjects are *kernel-microbench* ratios, **not** model-e2e; do not assume they translate (MEMORY: *validate-perf-headroom*; *accounting-delineates-what-matters*). Phase-2 must measure model-e2e per unit. |

---

## 8. Recommendation — is Phase 2 worth it, and at what scope?

**Yes, but scoped — do NOT bench all 24 units / 61 commits.**

1. **First (CPU, before any GPU):** re-baseline the +4.84% against `5e2ab3055de`. If a large chunk of it is actually the upstream inductor PRs (`_scaled_mm_v2`, atomic-add epilogue, flex vectorization), the landable-from-*this-branch* gain shrinks and changes everything.
2. **The branch is not "30+ independent landable commits."** It is **1 large entangled CORE PR** (mega-commit + ~30 co-evolving scheduler/codegen/cost-model commits, incl. the R1 fix) **+ ~11 optional independent FX-pass PRs**. Frame the landing that way.
3. **Spend GPU only on the few movers.** Use the Option-C codegen-diff proxy to rank, then bench ~6–10 targeted A/B pairs (~4–6 wall-hours). Expect the +4.84% to be carried by the **CORE scheduler-realization unit + 2–3 high-value passes**; units below the **0.82% A/A floor don't justify a standalone PR** (per the user's own bar).
4. **Land order:** CORE first (it's the foundation + carries R1); then the independent passes that each clear ~0.82% on their own; drop the rest (the 6 revert-pairs, 3 superseded, the 2× degenerate-dropout, the benchmarking-infra churn) — they contribute nothing or are infra-only.
5. **Big-spend flag:** the *full* per-unit walk is ~62 GPU-hours on 4×B200. The targeted plan is ~16–24. Recommend the targeted plan; get user sign-off before the walk.

---

### Appendix — provenance / method (all read-only)
- Range & topology: `git rev-list --count`, `git merge-base --is-ancestor`, `git log --first-parent`.
- Per-commit footprint: `git show --numstat --pretty=format:`.
- Survival: for each commit, `git show -U0` added product-lines vs `git show HEAD:<path>` content (multiset membership); revert-graph from `Revert "<subj>"` subject matching nearest prior identical subject.
- Reconstruction: union of surviving commits' product files vs `git diff 5e2ab..HEAD --name-only -- torch/_inductor value_ranges.py`.
- No `checkout`/`rebase`/`reset`/worktree was performed; HEAD and the owner WIP are untouched. Cherry-pick conflict-probing (Option B) is deferred to Phase 2 and must use a throwaway worktree.
