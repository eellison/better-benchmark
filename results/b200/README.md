# Perf results — 4x B200 (per-commit measurements)

> **Interim home.** These results live here, pinned to specific pytorch commits +
> hardware, until CI produces/versions them automatically. The pinned commits below
> are the source of truth for *what was measured*; numbers are valid only for
> exactly these commits.

## The model: a result is a MEASUREMENT at ONE commit; the A/B is DERIVED

A result here is a **measurement at one pytorch commit**: "the corpus kernels at
commit `C`, on hardware `H`, ran in `T` microseconds." Each measurement-set lives in
its own per-commit directory and carries exactly **one** `pytorch_commit` in its
`_metadata`.

The headline A/B — "how much did the perf branch improve models" — is **not a stored
artifact**. It is a **derivation over two measurement-sets**, computed on demand by
[`compute_ab.py`](compute_ab.py), which shape-matches the two commits' `kernels.json`
on `(pattern_hash, shape_hash)` and rolls up to per-model %-improvement. So there are
no `pytorch_baseline_commit`/`pytorch_branch_commit` dual fields anywhere — both
commits' measurements are stored as siblings, and the A/B is computed from them.

```
python compute_ab.py        # derives the HONEST headline (5e2ab base -> branch)
python compute_ab.py results/b200/5e2ab3055de results/b200/daa79cd25ca    # explicit (== default)
python compute_ab.py results/b200/244fdb379d11 results/b200/daa79cd25ca   # CONTAMINATED superset (see footnote)
```

## Provenance (what produced these numbers)

| | |
|---|---|
| **Hardware** | 4x NVIDIA B200 (SM100), GPU lock on, CUDAGraph replay, min-of-N |
| **pytorch BASELINE (D1, honest)** | `trunk@5e2ab3055de1bc4bffe2e7feffe1fc7ff7af8b10` (the perf branch's TRUE base — the commit it forked from; `pytorch_ref: "trunk"`). This is `compute_ab.py`'s default baseline. |
| **pytorch BASELINE (superset, contaminated — footnote only)** | `trunk@244fdb379d11d5925da5610b22e1466222c4afb9` (an *older* ancestor; **56 upstream pytorch-main PRs** sit between it and `5e2ab`, so `branch vs 244fdb` over-credits this branch with that upstream work — see footnote ‡). |
| **pytorch BRANCH (WIP perf work)** | `tmp_work@daa79cd25ca9a80bfd65799394cf4255d6be75a6` (`tmp_work` HEAD — the perf work under test; ref recorded as `pytorch_ref: "tmp_work"`) |
| **better-benchmark commit** | baseline run `3710b851` / branch run `a7f8b0af` (recorded as `bb_commit` in each file) |
| **swap set** | `torch/_inductor` + `torch/utils/_sympy/value_ranges.py` (Python-only, no rebuild; fresh `TORCHINDUCTOR_CACHE_DIR` per arm) |
| **corpus** | 1727 unique kernel patterns / 4977 (dir,shape) points; 158 non-genai models (73 train + 85 infer) + 8 genai microbenchmarks |
| **date** | 2026-06-18 |
| **harness** | dynamo-reset-per-shape fix `4ca6d532b` applied (see integrity notes) |

## Headline results (genai/microbenchmarks EXCLUDED — always cite this cut)

### Deliverable 1 — perf-branch model improvement (A/B), **DERIVED** via `compute_ab.py`
- **geomean +4.51% speedup** (≈1.045x; `compute_ab.py` prints the unrounded +4.510%),
  **median +2.18%**, **mean +4.05%** — 120 improved / 28 regressed / 10 flat, over 158
  non-genai models (range −20.0% .. +28.4%).
- Derived from pytorch `5e2ab3055de` (the branch's TRUE base) → `daa79cd25ca` (branch).
  **Run `compute_ab.py` to reproduce** (default arms) — it is not a stored number.
- **Largest regression: `torchbench/infer/pytorch_unet` −19.97%**; best improvement
  `torchbench/infer/pyhpc_isoneutral_mixing` +28.4%. `compute_ab.py` prints the full
  worst-8 / best-5 (genai-excluded) regression block.
- A/A noise floor ±0.82% → the win is real, above noise. Re-validated on the
  dynamo-reset-fixed harness; cache isolation + swap integrity confirmed
  (`REVALIDATION_SUMMARY.md`). Re-baselined to the correct base `5e2ab` 2026-06-19
  (`results/perf_ab/rebaseline_5e2ab_2026-06-19/`).

> ‡ **Footnote — the contaminated `244fdb` superset number.** The previously-cited
> headline used baseline `244fdb379d11`, giving **median +2.18% / mean +4.33% /
> geomean +4.84%** (n=158). That baseline is an *older* ancestor: **56 upstream
> pytorch-main PRs** (all `(#NNNNNN)`-tagged, touching `scheduler.py`/`lowering.py`)
> landed between `244fdb` and the perf branch's true base `5e2ab3055de`,
> independently of this branch. So `branch vs 244fdb` conflates this branch's work
> with that upstream work. **Re-baselining to `5e2ab` (the honest A/B above) leaves
> the median UNCHANGED (+2.18%) and trims the mean/geomean by ~0.3pp.** That entire
> ~0.3pp delta is **one upstream cross-entropy / NLL sum-reduction kernel**
> (`sum_81b4fd73f8d1`, ~2x faster upstream) mis-credited to just 2 train models
> (TrOCR +23%→+5%, Pegasus +21%→+5%). Per-point cross-baseline drift is at the A/A
> floor (signed +0.000%; only 2/4977 points exceed +30%, both that one kernel).
> `python compute_ab.py results/b200/244fdb379d11 results/b200/daa79cd25ca`
> reproduces the contaminated superset for cross-reference.

> **compile_us run-to-run variance caveat.** Individual per-kernel `compile_us`
> values (and therefore individual per-model A/B ratios and individual
> oracle-vs-compile ratios) carry run-to-run variance — e.g. on a capstone re-run
> the Longformer driver's compile dropped 785→637 µs (ratio 2.45→1.99) while its
> oracle floor was identical (320.3→320.5 µs). The **stable deliverable** is the
> **oracle FLOORS + the D2 ranking** (capstone median oracle ratio 0.9992); cite
> those as durable. Individual compile ratios are point-in-time.

### Deliverable 2 — distance to the reference-kernel (oracle) ceiling, **measured at the branch**
- **geomean +1.37%** potential speedup if every model hit its agent-kernel ceiling;
  **median 0.4%**, mean 1.3%. 24/152 models exactly at ceiling.
- Right-skewed: the corpus is mostly AT ceiling, with concentrated opportunity in a few
  models. Top: **AllenaiLongformerBase 37.4%** (one real, fixable kernel),
  swin 9.2%, vgg16 6.9%, BERT-train cluster ~4–5% (`DELIVERABLE2_v4_ranking.txt`).
- Two-floor accounting: **oracle-floor** (reference-kernel ceiling / optimization target)
  vs **min-floor** = min(oracle, compile) (achievable today). See `TWO_FLOOR_README.md`.

> **Geomeans** (D1 +4.51% vs the honest `5e2ab` base, D2 +1.37%) were **computed
> 2026-06-18/19 and are not yet emitted by a committed metric script.**
> `compute_ab.py` recomputes the D1 geomean on the fly and labels it as such; the
> median/mean are the long-standing committed cuts.

## Layout

```
results/b200/
  README.md                      <- you are here (entry point)
  compute_ab.py                  <- the A/B derivation (median/mean/geomean, genai-excluded)
  occurrences/                   <- per-model occurrence + extern-price sidecars (compute_ab.py input)
  CANONICAL_HEADLINE.md          <- the genai-excluded headline numbers (cite this)
  DELIVERABLE2_v4_ranking.txt    <- full per-model headroom ranking (distance to ceiling)
  TWO_FLOOR_README.md            <- the two-floor accounting explainer + top-15 under each
  REVALIDATION_SUMMARY.md        <- the D1 A/B re-validation (adversarial checks, cache isolation)
  longformer_HANDOFF.md          <- the #1 D2 result: root cause + fix design + handoff

  5e2ab3055de/                   <- BASELINE pytorch commit (HONEST D1 base; compute_ab.py default)
     kernels.json                   per-(dir,shape) compile-time measurements; _metadata.sweep_type=kernels

  244fdb379d11/                  <- contaminated SUPERSET baseline (older ancestor; footnote ‡ only)
     kernels.json                   kept for cross-reference; pass explicitly to compute_ab.py

  daa79cd25ca/                   <- BRANCH pytorch commit (measurement-set)
     kernels.json                   per-(dir,shape) measurements; sweep_type=kernels
     oracle_floors.json             v4 clean oracle floors (reference-kernel ceiling); sweep_type=oracle_floor
     oracle_floors_minfloor.json    min(oracle, compile) floors; sweep_type=oracle_floor
     projections.json               per-model absolute floor-time projection (oracle floor; coverage-gated)
     projections_minfloor.json      same, min floor (more models complete)
```

`oracle_floors*` and `projections*` exist only under the **branch** commit because that
is where the Deliverable-2 ceiling was swept; Deliverable 1 needs `kernels.json` from
**both** commits.

### `_metadata` schema (every JSON)

Each file carries exactly one `pytorch_commit` (the distinguishing axis) and a
`pytorch_ref` — the git branch/ref that commit was HEAD of at measurement time
(`tmp_work` for the branch arm, `trunk` for the two baselines). The ref disambiguates a
*moving* branch: `daa79cd25ca` is `tmp_work`'s HEAD today, but once `tmp_work`
advances that SHA is an orphan hash with no context, so the branch name is captured
as structured provenance (not just in the `harness_caveats` prose).

> **Note on the `5e2ab3055de` arm's `pytorch_commit`.** That baseline was measured
> via a Python-only inductor source swap (`git checkout 5e2ab -- torch/_inductor …`)
> while git HEAD stayed at `tmp_work`/`daa79`, so the bench tool originally
> mis-stamped `pytorch_commit=daa79`/`pytorch_ref=work2`. Its `_metadata` here is
> corrected to the *actually-measured* sources (`pytorch_commit=5e2ab3055de`,
> `pytorch_ref=trunk`); `harness_caveats` records the swap + the original mis-stamp,
> and `source_artifact` points at the raw arm in
> `results/perf_ab/rebaseline_5e2ab_2026-06-19/`.

Also present: the
`bb_commit` (the better-benchmark commit that ran it — the old bare `commit` field,
renamed and preserved), `hardware`, `date`, `sweep_type` (`kernels` | `oracle_floor` |
`projection`), `corpus_size`, and `harness_caveats`. The kernel files also retain their
original bench-tool fields (`tool`, `timestamp`, `n_repros`, …) for provenance.

The two `projections*.json` were **bare JSON lists** of model rows in the source; they
are wrapped here as `{"_metadata": {...}, "models": [...]}` so they can carry metadata
(noted in each file's `_metadata.wrapping_note`).

## Key integrity notes (why these numbers are trustworthy)

- **Harness contamination fixed** (`4ca6d532b`): `bench_oracle` now resets dynamo per
  shape. Without it, multi-shape oracle benches recompiled DYNAMIC on later shapes →
  CUDAGraph-captured at the wrong specialization → compile_us inflated 10–46x. Caught
  from a suspicious var_mean "46x" and traced to a positional fingerprint (0/393
  first-shape vs 265/2676 later-shape ≥5x). The v4 sweep re-measured all multi-shape dirs
  clean (265→0 ≥5x points).
- **Bench-hacks removed**: 11 `f5be`-class tolerance-self-blend oracles found via source
  audit (the `--check` gate can't catch them) and rewritten faithfully.
- **genai-EXCLUDED** in the headline: the 8 genai microbenchmarks are raw kernel-level
  wins on directly-targeted ops and distort the model-level aggregate; `compute_ab.py`
  reports them under a separate, clearly-labeled MICROBENCHMARKS block and **never folds
  them into the headline mean/median/geomean**. They are much larger
  (`CrossEntropyForward +66.16%`, `SoftmaxForward +40.19%`, `CrossEntropyBackward
  +41.62%`, …). `compute_ab.py` also prints a genai-**included** reference line
  (against the honest `5e2ab` base: median +2.46% / mean +5.04% / geomean +6.02%,
  n=166) explicitly marked NOT the headline.
- **Same extern both arms**: externs (cuBLAS/cuDNN/flash) are inductor-invariant, so the
  *same* extern price feeds both arms (all 3937 priced extern entries have
  `baseline_us == branch_us`). It enters only the A/B denominator and cancels in the
  numerator — `compute_ab.py` inherits this from `rollup.py`. The occurrence sidecars under
  `occurrences/` are shared across commits, not per-arm.

## Caveat / status

This is a manual snapshot pinned to a WIP pytorch branch (`tmp_work`, not yet merged).
Numbers will shift as that branch evolves; re-derive against the pinned commits to
reproduce. The intended end state is CI-produced, versioned results — this directory is
the interim, human-curated stand-in.
