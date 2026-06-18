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
python compute_ab.py        # derives the headline from the two dirs below
python compute_ab.py results/b200/244fdb379d11 results/b200/daa79cd25ca   # explicit
```

## Provenance (what produced these numbers)

| | |
|---|---|
| **Hardware** | 4x NVIDIA B200 (SM100), GPU lock on, CUDAGraph replay, min-of-N |
| **pytorch BASELINE** | `244fdb379d11d5925da5610b22e1466222c4afb9` (merge-base / ancestor of the perf branch) |
| **pytorch BRANCH (WIP perf work)** | `daa79cd25ca9a80bfd65799394cf4255d6be75a6` (`tmp_work` HEAD — the perf work under test) |
| **better-benchmark commit** | baseline run `3710b851` / branch run `a7f8b0af` (recorded as `bb_commit` in each file) |
| **swap set** | `torch/_inductor` + `torch/utils/_sympy/value_ranges.py` (Python-only, no rebuild; fresh `TORCHINDUCTOR_CACHE_DIR` per arm) |
| **corpus** | 1727 unique kernel patterns / 4977 (dir,shape) points; 158 non-genai models (73 train + 85 infer) + 8 genai microbenchmarks |
| **date** | 2026-06-18 |
| **harness** | dynamo-reset-per-shape fix `4ca6d532b` applied (see integrity notes) |

## Headline results (genai/microbenchmarks EXCLUDED — always cite this cut)

### Deliverable 1 — perf-branch model improvement (A/B), **DERIVED** via `compute_ab.py`
- **geomean +4.83% speedup** (≈1.048x; `compute_ab.py` prints the unrounded +4.838%),
  **median +2.18%**, **mean +4.33%** — 119 improved / 31 regressed / 8 flat, over 158
  non-genai models (range −20.0% .. +28.0%).
- Derived from pytorch `244fdb379d11` (baseline) → `daa79cd25ca` (branch). **Run
  `compute_ab.py` to reproduce** — it is not a stored number.
- **Largest regression: `torchbench/infer/pytorch_unet` −19.97%**; best improvement
  `torchbench/infer/pyhpc_isoneutral_mixing` +27.97%. `compute_ab.py` prints the full
  worst-8 / best-5 (genai-excluded) regression block.
- A/A noise floor ±0.82% → the win is real, above noise. Re-validated on the
  dynamo-reset-fixed harness; cache isolation + swap integrity confirmed
  (`REVALIDATION_SUMMARY.md`).

### Deliverable 2 — distance to the reference-kernel (oracle) ceiling, **measured at the branch**
- **geomean +1.37%** potential speedup if every model hit its agent-kernel ceiling;
  **median 0.4%**, mean 1.3%. 24/152 models exactly at ceiling.
- Right-skewed: the corpus is mostly AT ceiling, with concentrated opportunity in a few
  models. Top: **AllenaiLongformerBase 37.4%** (one real, fixable kernel),
  swin 9.2%, vgg16 6.9%, BERT-train cluster ~4–5% (`DELIVERABLE2_v4_ranking.txt`).
- Two-floor accounting: **oracle-floor** (reference-kernel ceiling / optimization target)
  vs **min-floor** = min(oracle, compile) (achievable today). See `TWO_FLOOR_README.md`.

> **Geomeans** (D1 +4.83%, D2 +1.37%) were **computed 2026-06-18 and are not yet emitted
> by a committed metric script.** `compute_ab.py` recomputes the D1 geomean on the fly and
> labels it as such; the median/mean are the long-standing committed cuts.

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

  244fdb379d11/                  <- BASELINE pytorch commit (measurement-set)
     kernels.json                   per-(dir,shape) compile-time measurements; _metadata.sweep_type=kernels

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

Each file carries exactly one `pytorch_commit` (the distinguishing axis), the
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
  (median +2.30% / mean +5.31% / geomean +6.33%, n=166) explicitly marked NOT the headline.
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
