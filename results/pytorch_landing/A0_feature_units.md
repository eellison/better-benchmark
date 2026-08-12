# PHASE-2 STEP A0 — PyTorch perf branch: REAL landable feature-units

**Date:** 2026-06-22 · **Mode:** read-only CPU archaeology (no GPU, no checkout, no worktree — pure `git log/show/diff/grep`; `/tmp/pytorch-work` HEAD + WIP untouched).
**A/B range:** A = `5e2ab3055de` … B = `daa79cd25ca` (83 commits). The `244fdb..5e2ab` gap = 56 UPSTREAM PRs, **excluded**.
**Net product diff:** `git diff 5e2ab3055de daa79cd25ca -- torch/_inductor torch/utils/_sympy/value_ranges.py` = **36 files, +13,821 / −200**.
**Companion machine-readable file:** `A0_feature_units.json` (one record per unit, this directory).

---

## 0. Headline numbers (report-back)

| Quantity | Count |
|---|---|
| **Feature-units** (distinct coherent optimizations) | **33** |
| …contributing non-zero at HEAD | **29** (4 are dead/net-zero) |
| **FLAG-TOGGLEABLE** units (clean A/B via a config flag) | **24** |
| …of which use an env var (`TORCHINDUCTOR_*`) | 18 |
| …of which are a python-bool config attr (config-patch toggle) | 6 |
| **NEEDS selective-revert / cumulative-only** | **5** (U30 MOR, U31 CE, U32 argmax, no-flag parts of U33, U25 fix-chain) |
| **DEAD / net-zero** (do not land) | **4** |
| **Cleanly independent PRs** (new file + own flag) | **~17–19** |
| **Entangled-in-core** units | **12** (but 7 of those are still flag-toggleable for A/B) |
| **A1 sweep estimate** | **~30 A/B arms naive; ~6–10 after proxy-prune** (see §5) |

**Corrected count vs Phase-1:** Phase-1 said *"1 CORE + ~24 units (11 separable + 1 entangled core)"* and framed the core as **cumulative-only**. That under-counts both the unit count (**33**, not ~24) **and the toggleability** — Phase-1 lumped evict_first / scalar_acc / online-softmax / split-reduction / realize-on-reuse into one cumulative blob, but **each has its own config flag** and is cleanly A/B-able. The truly cumulative-only surface is only **5** items, not the whole core.

---

## 1. Corrections to Phase-1 (the under-count)

1. **`scatter_reduce_fusion.py` is not one unit — it hosts 4 separately-flagged passes** (U01–U04): `scatter_reduce_fusion`, `scatter_add_reduce_elimination`, `scatter_add_into_fusion`(+chained), `reduce_scatter_distribution`. Each has a distinct `config.*` gate and is registered separately in `post_grad.py` (lines 656/662/668/674).
2. **The "CORE triton-codegen unit" is flag-toggleable, not cumulative.** `evict_first_requires_aligned_strides`, `online_softmax_fast_combine`, `scalar_reduction_accumulators`, `scalar_acc_configs_without_cd` are all `config.triton.*` / `config.*` flags (U27/U28/U29). A/B each by toggling its flag on `B=HEAD`.
3. **The "split-reduction cost model" is 3 flagged units** (U22/U23/U24): `split_reductions_for_{undersaturated,partially_saturated}_gpu`, `segment_aligned_split_reductions`, `split_multilayer_second_step`, `split_reductions_for_fusion`.
4. **The scheduler-realization family splits into U25 (producer-inlining, flagged) + U26 (ir realize-on-reuse, flagged).** Only the *fix-chain* glue in U25 is cumulative.
5. **`reduction_chaining.py` (216 L) is DEAD CODE at HEAD** — never imported or called anywhere in the tree; its gate `config.triton.reduction_chaining` isn't even in the config diff (reads a missing attr → False). **Zero contribution. Do not land.** (Phase-1 listed it as a CORE feature.)
6. **CE gather-into-softmax is a real *new fused reduction* (U31)** — a new prim `cross_entropy_loss_online` (`inductor_prims.py +25`), a new reduction type `online_softmax_cross_entropy` (`ops_handler.py +1`), plus `lowering.py` + `codegen/triton.py`. **No flag → selective revert.**
7. **MOR Triton finalize (U30)** = `simd.py::_emit_triton_finalize_sum (+176)`, **no flag → selective revert** (mega-commit claims "35% Swin"). The sibling-fusion *enable* path is gated by `config._sibling_reduction_fusion`, which is **False at HEAD**.
8. **`reduced_atomic_contention.py` / `partitioned_scatter_*` exist at baseline `5e2ab`** → UPSTREAM, correctly excluded.
9. **Net-zero (excluded):** degenerate-dropout (added+reverted ×2), smart-realize_hint (added+reverted), num_stages tune / INNER-threshold 1024→8192 / realize_reads 4→30 (each reverted). NB `memory.py +28` survives from the mislabeled revert `f74a7f13723` and attaches to U25.

---

## 2. The 33 feature-units

Legend — Benching: **F** = FLAG-TOGGLEABLE (env or bool) · **R** = selective-revert / cumulative · **P** = needs-probe. Dep: **ind** = independent · **core** = shares a core file (still F if it has a flag).

### A. Independent FX-pass units (each ≈ one PR)

| # | Unit | Owning file(s) (LOC) | Flag | Bench | Self-reported impact (UNVERIFIED microbench) |
|---|---|---|---|---|---|
| U01 | scatter_reduce_fusion (masked/bilinear) | `fx_passes/scatter_reduce_fusion.py` 2750 + `kernel/scatter_reduce_gather.py` 299 | `scatter_reduce_fusion` (env, **default OFF**) | F | up to 3.60×→0.45x |
| U02 | scatter_add → reduce elimination | (same file: `scatter_add_gather_reduce_pass`) | `scatter_add_reduce_elimination` (env, 1) | F | 3.60×→0.45x sum_sum_3219a09ab96a |
| U03 | scatter_add_into fusion (+chained) | (same file) | `scatter_add_into_fusion`(+`_chained`) bool | F | embed-bwd; chained 1.39×→0.82x; dtype 2.18×→0.87x |
| U04 | reduce_scatter distribution (LN-bwd) | (same file) | `reduce_scatter_distribution` bool | F | 1.97×→1.93x (**barely moved — weak**) |
| U05 | structured_scatter_decode (affine-iota overlap-add) | `fx_passes/structured_scatter_decode.py` 413 | `structured_scatter_decode` (env, 1) | F | 2.85×→1.47x |
| U06 | select_scatter_sparsity | `fx_passes/select_scatter_sparsity.py` 895 | `select_scatter_sparsity` (env, 1) | F | topo-fix only |
| U07 | slice_scatter_elision (+partial) | `fx_passes/slice_scatter_elision.py` 433 | `slice_scatter_elision`(+`_partial`) (env, 1) | F | — |
| U08 | as_strided_scatter_elision | `fx_passes/as_strided_scatter_elision.py` 282 | `as_strided_scatter_elision` (env, 1) | F | — (mega-seeded only) |
| U09 | linear_reduction_elimination | `fx_passes/linear_reduction_elimination.py` 726 | `linear_reduction_elimination` (env, 1) | F | — (mega-seeded only) |
| U10 | layout_transform_store_sinking | `fx_passes/layout_transform_store_sinking.py` 260 | `layout_transform_store_sinking` (env, 1) | F | — (mega-seeded only) |
| U11 | diagonal_skew_elimination (Longformer) | `fx_passes/diagonal_skew_elimination.py` 461 + `scatter_reduce_fusion.py` +11/−2 shim | `diagonal_skew_elimination` (env, 1) | F | 8→5 kernels — **R3 Longformer −9.4% fix** |
| U12 | one_hot_reduction_elimination | `fx_passes/one_hot_reduction_elimination.py` 492 | `one_hot_reduction_elimination` (env, 1) | F | 1.62×→1.03x |
| U13 | expand_sum_elision | `fx_passes/expand_sum_elision.py` 370 | `expand_sum_elision` (env, 1) | F | 1.694×→1.17x |
| U14 | cat_through_reduction | `fx_passes/cat_through_reduction.py` 281 | `cat_through_reduction` (env, 1) | F | 1.81×→0.90x; 1.45×→0.61x |
| U15 | rotate_half_gather (RoPE) | `fx_passes/rotate_half_gather.py` 390 | `rotate_half_gather` (env, 1) | F | 1.27×→1.00x |
| U16 | index_through_norm | `fx_passes/index_through_norm.py` 546 | `index_through_norm` (env, 1) | F | — (bundled in afc3) |
| U17 | bn_affine_folding | `fx_passes/bn_affine_folding.py` 380 | `fold_bn_affine` (env, 1) | F | 1.27×→1.23x (weak) |
| U18 | dedupe_graph_outputs | `fx_passes/dedupe_graph_outputs.py` 227 | `dedupe_graph_outputs` (env, 1) | F | 5.13×→0.71x (verify e2e) |
| U19 | constant-fold uniform-thru-view + iota/arange | `fx_passes/joint_graph.py` +103 (ConstantFolder) | **no dedicated flag** (rides joint-graph const-fold) | P | 1.44×→1.0x; 1.098→1.092 (~noise) |
| U20 | rsqrt_canonicalization | `fx_passes/post_grad.py` (~L2166) | `rsqrt_canonicalization` (env, 1) | F | 1.41×→0.99x |
| U21 | slice_output_compaction | `fx_passes/slice_output_compaction.py` 220 | `compact_slice_outputs` (env, 1) | F | 1.87×→1.00x |

### B. Core-codegen units WITH a flag (entangled file, still A/B-able)

| # | Unit | Owning file(s) | Flag | Bench | Self-reported (UNVERIFIED) |
|---|---|---|---|---|---|
| U22 | split-reduction: under/partially-saturated GPU | `choices.py +284`, `config.py` | `split_reductions_for_{undersaturated,partially_saturated}_gpu` bool | F | 1.82×→1.29x; 2.96×→0.69x |
| U23 | segment-aligned split introduction | `choices.py`,`ir.py` | `segment_aligned_split_reductions` (env, 1) | F | 1.41×→0.62x; 1.26×→0.89x |
| U24 | multilayer-2nd-step split + split-for-fusion | `ir.py`,`scheduler.py` (L4171) | `split_multilayer_second_step`+`split_reductions_for_fusion` bool | F | 1.88×→0.77x |
| U25 | scheduler `inline_recomputable_producers` + cheap-into-stencils (**R1 pytorch_unet family**) | `scheduler.py +1116`, `lowering.py` (L5670), `memory.py +28` | `inline_recomputable_producers`+`inline_cheap_producers_into_stencils` bool | F* | 1.43×→0.75x; 1.41×→1.10x. *Flag toggles inlining, but the 8-commit fix-chain (dead-node-elim, mutation_renames, ExternKernel-read fixes) is integral → bench as ONE cumulative unit. R1 −20% **and** its fix both live here. |
| U26 | ir realize-on-reuse: transcendentals + indirect-index | `ir.py` (L10265, L10282) | `realize_heavy_transcendentals_on_reuse`+`realize_indirect_indexing_on_reuse` bool | F | 2.19×→1.27x; 1.62×→0.89x |
| U27 | triton evict_first policy | `codegen/triton.py` (~L4164) | `config.triton.evict_first_requires_aligned_strides` bool | F | 1.42×→0.98x (+ counterexample gates) |
| U28 | online-softmax fast combine | `codegen/triton.py` (L4622+) | `config.triton.online_softmax_fast_combine` | F | 1.42×→1.01x |
| U29 | scalar reduction accumulators (scalar_acc) | `codegen/triton.py` (L5041+), `runtime/triton_heuristics.py` (L4049) | `config.triton.scalar_reduction_accumulators` + `scalar_acc_configs_without_cd` (env) | F | genai SoftmaxForward 3541→1905us |

### C. Core units WITH NO flag → selective revert / cumulative

| # | Unit | Owning file(s) | Bench | Self-reported (UNVERIFIED) |
|---|---|---|---|---|
| U30 | **MOR (multi-output-reduction) Triton finalize** | `codegen/simd.py::_emit_triton_finalize_sum +176`; sibling-enable in `ir.py` gated by `_sibling_reduction_fusion`=**False@HEAD** | R/P | **"35% Swin"** (mega msg); 2.02×→1.02x |
| U31 | **CE gather-into-softmax** (new fused reduction) | `inductor_prims.py +25` (new prim), `ops_handler.py +1`, `lowering.py` (L9177), `codegen/triton.py` | R | mega "0.8%" |
| U32 | argmax/argmin tie-break elimination | `ir.py` reduction-combine table (L1219–1640) | R | 1.59×→1.31x |
| U33 | triton_heuristics: persistent-thr raise (Blackwell BN) + low-warp + cooperative-widen + rnumel-ceiling fix | `runtime/triton_heuristics.py +145`, `choices.py` (cooperative), `config.py` (`persistent_reduction_threshold_inner` env scalar) | P/R | 1.32×→1.0x; 1.54×→1.25x |

### Loose / micro-units (could be promoted, tiny)
- **`elide_constant_index_asserts`** (`bounds.py +52`, `lowering.py`, `value_ranges.py +9`) — flag `elide_constant_index_asserts` bool; bundled in afc3 with U16.
- **`clamp_embedding_indices`** (`lowering.py` L4443) — flag `clamp_embedding_indices` bool; 1.52×→1.01x (commit `aa83bc1951c`). Could be **U34**.
- **`prefer_concat_kernel_shared_reads_threshold=6`** (`lowering.py` L2295, SwiGLU pointwise_cat) — scalar flag (set high to disable); 2.10×→0.97x. Could be **U35**.
- **`hoist_invariant_compute`** (`triton.py` L3954/3981) — flag `hoist_invariant_compute` env **default OFF** → **zero contribution at HEAD** (author: "neutral on B200").

### DEAD / net-zero (do NOT land)
- **`reduction_chaining.py`** (216 L) — orphaned, never called, gate missing. Zero.
- Degenerate-dropout pass (added+reverted ×2); smart-realize_hint (added+reverted); num_stages tune / INNER-thr 1024→8192 / realize_reads 4→30 (each reverted).

---

## 3. Flag-toggleable vs cumulative split

- **24 FLAG-TOGGLEABLE** (clean A/B by flipping one flag on `B=HEAD`): U01–U18, U20, U21 (passes) + U22, U23, U24, U26, U27, U28, U29 (flagged core). Of these, **18 use a `TORCHINDUCTOR_*` env var** (purest A/B — no rebuild), **6 are python-bool config attrs** (toggle in a config patch).
  - ⚠ Two flag-toggleable units default **OFF** at HEAD: **U01 `scatter_reduce_fusion`** (env default 0) and **hoist_invariant_compute** (default 0). For these, "A/B" means flag-ON vs HEAD-default — and hoist contributes **zero** as shipped.
- **U25** is *partly* toggleable (the inlining bool) but its fix-chain is integral → bench **cumulatively** as one unit.
- **5 cumulative-only / selective-revert:** U30 (MOR finalize), U31 (CE fused reduction), U32 (argmax combine-fn), U33 (heuristics, mostly no-flag), and U25's fix-chain.

**Independent PRs:** **~17 cleanly independent** (U01–U10, U12–U18, U20, U21 — new file + own env flag, only textual coupling via the `post_grad.py` registry touched by ~20 commits). Add **U11** (independent + 11-line `scatter_reduce_fusion.py` shim → land after U01) and **U19** (no flag, joint_graph hook) → **~19 independent-capable**. The remaining **12 are entangled in core files** (`choices`/`scheduler`/`ir`/`triton`/`simd`/`heuristics`), though 7 of those stay flag-A/B-able.

---

## 4. The mega-commit anchor

`97385fb3273` (5,237 ins / 17 files) **seeds**: U01 (+ kernel file), U08, U09, U10, U25 (scheduler machinery), U26 (partial), U29 (partial), U30 (simd finalize), U31 (CE prim/handler), U33 (partial) **and** the dead `reduction_chaining.py`. The independent passes seeded here (U08/U09/U10) can be re-extracted as standalone files; the core hooks (U25/U26/U29/U30/U31/U33) cannot be cherry-picked apart from it. **Land (or replicate) the mega-commit's core hooks first**, then the entangled units; the standalone passes (U01–U18 etc.) are independent of it except U11's shim.

---

## 5. A1 sweep-count estimate

Assume `A=B=HEAD` (`daa79cd25ca`) is the shared reference and each flag-toggleable unit needs **one extra arm** = HEAD with that single flag flipped (env-flag flips need no rebuild; bool flips need a config patch, no recompile of C++).

| Bucket | Arms |
|---|---|
| Baseline `5e2ab` (full-off reference) + HEAD (full-on) | 2 |
| 24 flag-toggleable units × 1 single-flag-off arm | 24 |
| U25 cumulative (1 checkout-state arm) | 1 |
| 4 selective-revert arms (U30, U31, U32, U33) | 4 |
| **Naive total distinct arms** | **~31** |

At Phase-1's ~36 min/arm on 4×B200, the naive full set ≈ **~18.5 wall-hours**. **Recommendation (matches Phase-1's Option C):** run the **CPU codegen-diff proxy** first — for each unit, compile the affected repro set with the flag flipped and diff the emitted Triton; only units that change generated kernels can move e2e perf. Expectation: the +4.84% geomean is carried by **a handful** — the R1 realization family (U25/U26), MOR (U30), and 2–3 high-multiplier passes (U18 dedupe 5.13×, U05 structured-scatter 2.85×, the scatter U01–U04 family) — so the GPU spend prunes to **~6–10 targeted A/B arms ≈ 4–6 wall-hours**. Units whose single-flag-off arm is within the ~0.82% A/A noise floor don't justify a standalone PR.

**Single-flag-off is the cleanest A1 design**: it isolates each unit's *marginal* contribution at the fully-optimized operating point (avoids the order-dependence of an additive cumulative walk), and 18 of the 24 toggles need no rebuild at all.

---

## 6. Reconstruction cross-check

All **36** net-diff files are assigned: 28 to units U01–U33 directly; `runtime/benchmarking.py (+47)` is **plumbing-only** (5 `Update` commits, autotune-timing infra, ~0% perf — not a unit); `codegen/common.py (+18)` and `utils.py (+10)` are **support glue** for U30/hoist. **No residual product change is unassigned.** Reconstruction complete.

---

### Provenance (all read-only)
`git log --oneline 5e2ab..daa79cd`, `git diff --stat 5e2ab..daa79cd -- torch/_inductor value_ranges.py`, `git log --oneline 5e2ab..daa79cd -- <file>` (per-file attribution), `git show <sha>:torch/_inductor/fx_passes/post_grad.py | grep config.*` (flag→pass registry), `git diff 5e2ab..daa79cd -- config.py | grep` (flag inventory), `git grep config.<flag> daa79cd -- torch/_inductor` (gating sites), `git show 5e2ab:.../reduced_atomic_contention.py` (baseline-existence → upstream). No checkout/rebase/reset/worktree; HEAD + owner WIP untouched.
