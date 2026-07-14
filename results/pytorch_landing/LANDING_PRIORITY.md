# Prioritized landing list — per-commit, kernel-geomean + model-e2e geomean

**Date:** 2026-06-23 · **Base:** `5e2ab3055de` → **HEAD:** `daa79cd25ca` · genai-excluded, n=158.
**Two metrics per row:**
- **kernel Δ** = geomean over per-kernel ratios (`FULL_per_commit_table.txt`). Tighter floor (~0.2%, measured).
- **model Δ** = per-model-e2e geomean, fusible+extern denominator (`FULL_per_commit_e2e.txt`). The HONEST model-impact; kernel-geomean overstates ~2-3× due to extern dilution.

> **NOT additive.** Each Δ is measured vs the previous cumulative state (moving baseline), so the column does NOT sum to the branch total. Whole branch = **+2.18% median / +4.51% geomean** model-e2e. These rows RANK contribution; they don't add.
> **Floors:** kernel-geomean ±0.2% · model-e2e ±0.82pp (model A/A). A row below the model floor is "real kernel win, sub-floor at model level" — BUNDLE, don't headline.

---

## TIER 0 — LAND FIRST (headline, unambiguous)

| # | Commit | Unit | Feature | kernel Δ | model Δ | Note |
|---|--------|------|---------|---------:|--------:|------|
| 1 | `a73d1583b34` | U20 | rsqrt canonicalization | **+5.64pp** | **+2.11pp** | Verified. ~half the branch's model gain. 45-line peephole. Concentrated on conv/BN models (~1.15× per affected model, ~0 elsewhere). |

## TIER 1 — LAND (clean, above-or-near floor, own mechanism)

| # | Commit | Unit | Feature | kernel Δ | model Δ | Note |
|---|--------|------|---------|---------:|--------:|------|
| 2 | `ab29345d82d` | U31 | CE loop-invariant hoist | **+1.22pp** | **+0.54pp** | Above model floor. Selective-revert PR (carries online_softmax_cross_entropy prim). |
| 3 | `a26fc2c8bf4` | U28 | online-softmax fast combine | +0.83pp | +0.33pp | Own flag. Big genai lever too (Softmax +10%, CE +20%). |
| 4 | `ca8f961d6b0`+`9dde2c59a51` | U29 | scalar_acc + num_load≤4 gate | +0.59 / +0.44pp | +0.18 / +0.06pp | Gate is a regression-FIXER. THE genai lever (Softmax +46%, CE +64%). Ship the two together. |
| 5 | **U10** (in `97385fb3273`) | U10 | layout_transform_store_sinking | **+1.20× on its 14 kernels** | small (ShuffleNet-only) | Only clean win the mega-ablation found. Structural (nk 1→3). LAND clean; verify per-model on ShuffleNet rollup. |

## TIER 2 — LAND-WITH-GATE (real win, ships a regression unless gated)

| # | Commit | Unit | Feature | kernel Δ | model Δ | Gate required |
|---|--------|------|---------|---------:|--------:|---------------|
| 6 | `d8e9914094a` | U11 | diagonal_skew (Longformer) | +0.18pp | +0.03pp | **R3 patch** (validated). Without it: Longformer −9.4% (the #1 model). |
| 7 | `905450a5a5d`+chain | U25 | inline_recomputable_producers | +0.29pp (span) | +0.20pp (span) | **REWORK — R1 is NOT sufficient (RESOLVED 2026-07-14).** Measured: R1 fixes ONLY the unet stencil class (737→217us, 3.4x) and preserves the Longformer win, but class-B transformer regressions (XLNet/XGLM/GPTNeo, 1.33-1.43x vs pass-off) are BYTE-IDENTICAL codegen under R1 — its extra_loads term is mechanically 0 for stencil_reads=1 inlines. Needs two additional cost-model terms: (1) split memory_saved across multi-consumers, (2) price producer re-loads per r0 pass when inlining flips a persistent reduction to looped. Handoff: /tmp/scratch_space/r1_transformer_check/VERDICT.json. Do NOT land U25 with R1 alone. |

## TIER 3 — BUNDLE (real but sub-floor; ship inside a family PR, never headline)

| Commit | Unit | Feature | kernel Δ | model Δ |
|--------|------|---------|---------:|--------:|
| `f595e0cac3a` | U06 | select_scatter_sparsity topo-fix | +0.50pp | +0.15pp |
| `1406552b9d3` | U15 | rotate_half_gather (RoPE) | +0.37pp | +0.08pp |
| `4427bfd553c` (span) | U03 | scatter_add_into_fusion dtype-cast | +0.35pp | +0.16pp |
| `a85d79a900a` | U23 | segment-aligned split | +0.25pp | +0.16pp |
| `56959375c33` | U05 | structured_scatter_decode | +0.13pp | +0.01pp |
| U07/U08 (in mega) | U07/U08 | scatter-elision | ~floor | negligible | (gate U08's tf_efficientnet 0.75× regression) |

## DROP — do not open PRs

| Item | Commit | kernel Δ | model Δ | Reason |
|------|--------|---------:|--------:|--------|
| BN-affine-fold | `86c5451e197` | **−0.76pp** | **−0.29pp** | Net-negative on corpus. |
| linear_reduction_elim (U09) | in mega | **0.9929 (net-neg)** | ~0 | Flag-off is faster; no structural gate. |
| scatter_reduce_fusion (U01) | in mega | n/a | n/a | Corpus-dead: 0/1727 fire even forced ON. |
| realize_reads_threshold 4→30 | `1a824e0747a` / rev `f74a7f13723` | +0.83 / −0.82 | +0.41 / −0.37 | Added + reverted → net ≈ 0. |
| persistent-reduction INNER thresh | `d2cd52d3117` / rev `62f27911a59` | +0.14 / −0.24 | +0.04 / −0.05 | Added + reverted → net ≤ 0. |
| num_stages tune | rev `f0b54e1aa1f` | +0.35 (revert) | +0.12 | Net-zero pair; revert is the end-state. |
| reduction_chaining.py | in mega | 0 | 0 | Dead code, never called. |

## SELECTIVE-REVERT TO ATTRIBUTE (no flag — bulk of mega's claimed +1.95pp/+0.71pp lives here)
- **U30** MOR-finalize ("35% Swin" — bench against Swin rollup)
- **U31** CE-prim (bench against CrossEntropy genai micros)
- **U33** Blackwell-BN + low-warp configs
These can't be flag-ablated; their isolated impact is UNMEASURED. The mega's model-e2e +0.71pp is mostly here (only U10 of the 6 flag-units is a clean positive; U09/U25 net-negative).

---

### One-line summary
**Land rsqrt (≈half the model win) + 4 clean Tier-1 PRs (CE-hoist, online-softmax, scalar_acc, U10) + U30 MOR-finalize, then 1 gated PR (diagonal-skew+R3).** Inline (U25) moved to REWORK — the R1-covers-transformers check (2026-07-14) proved R1 fixes only the stencil class; U25 needs two more cost-model terms before it's landable. Tier-3 bundles ride along inside family PRs. Drop the 7 net-neg/dead/reverted items. (U30/U31/U33 were resolved by selective-revert: U30 lands, U31 drops, U33 is separate commits — see A2 §7.)
