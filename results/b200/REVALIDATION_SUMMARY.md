# Deliverable 1 re-validation — clean A/B re-run (2026-06-18)

**Verdict: REPRODUCES.** The perf-branch model win reproduces within the A/A
noise floor after the dynamo-reset harness change (commit `4ca6d532b`). That
change did **NOT** affect the A/B measurement.

## Commits used (verified)

| Arm | Commit | vs prompt's expected |
|---|---|---|
| Branch | `daa79cd25ca` (tmp_work HEAD) | MATCHES `daa79cd25ca` |
| Baseline | `244fdb379d11` (ancestor of tmp_work; the documented merge-base) | MATCHES `244fdb379d11` |
| Corpus | `repros/canonical` @ HEAD `3710b851589` (branch investigations-june-2026) | 1727 dirs / 4977 shape points |
| Swap set | `torch/_inductor` + `torch/utils/_sympy/value_ranges.py` | as documented |

Note: `git merge-base tmp_work main` reports `0f1438d5f85c`, but `244fdb379d11`
is an ancestor of tmp_work and is the baseline the harness (`run_arm.sh`) and
prior rollup pinned. We used exactly `244fdb379d11` for baseline.

Pre-run hygiene: two uncommitted files in /tmp/pytorch-work
(`torch/_inductor/config.py`, `fx_passes/scatter_reduce_fusion.py` — the parked
scatter_add regression experiment) were stashed before the sweep and restored
after, so neither arm was contaminated by uncommitted work. Each arm's
`git checkout <commit> -- torch/_inductor ...` set the tree to the committed
state of that commit.

## (b) Recomputed Deliverable 1 — does it reproduce +2.34% / +4.41%?

Per-model `pct_improvement` via the EXACT `rollup.py` arithmetic (fusible
(phash,shape) join, coord_descent_us pref, rep-median fallback, extern
denominator from occurrence sidecars), over the unetfixed-class arms (the corpus
carries FAITHFUL pytorch_unet indices `gen=['index',0,320/479]`, so a fresh
bench reproduces the unetfixed numbers, NOT the degenerate pre-fix ones).

| Cut | NEW (revalidate) | PRIOR (live rollup.json) | Δ median | Δ mean |
|---|---|---|---|---|
| genai-incl (166) | median **+2.303%** / mean **+5.313%** / 126-31-9 | median +2.343% / mean +5.278% / 123-36-7 | **-0.04pp** | +0.03pp |
| genai-excl (158) | median **+2.179%** / mean **+4.326%** / 119-31-8 | median +2.228% / mean +4.328% / 117-34-7 | **-0.05pp** | -0.00pp |

The prompt's "+2.34% median / +4.41% mean, 112/33/7, 152 models" is a recalled
blend of cuts: +2.34% = the genai-incl median; +4.41% ≈ the genai-excl mean
(+4.33%). No single committed file holds exactly 112/33/7 over 152; the
canonical durable numbers are median +2.23–2.34% / mean +4.33–5.31% / ~117-34-7
depending on genai-incl/excl. **The re-run lands on top of those, all within the
±0.82% A/A band.** (The "152-model" figure is the oracle-HEADROOM set, a
different join, not the A/B-improvement set which is 158 non-genai / 166 total.)

Median Δ |−0.04pp| (incl) / |−0.05pp| (excl) ≪ ±0.82%. **Reproduces.**

## (c) Did the dynamo-reset change (4ca6d532b) affect A/B numbers?

**NO.** Evidence:

1. **Code path is byte-identical.** The A/B drives
   `bench_parallel.py repros/canonical --all-shapes` (no `--oracles`), which hits
   `bench_one`. That function ALREADY does `torch._dynamo.reset()` per shape,
   before each compile (bench_parallel.py:2440 default arm, :2453 cd arm, inside
   the per-shape loop at :2419). Commit `4ca6d532b` touched ONLY
   `oracle_harness.py` (+9 lines) — its own message says it "Mirror[s] the repro
   path (bench_parallel.py)", i.e. the fix was COPIED FROM the path the A/B uses.
   The only other post-prior-A/B change to bench_parallel.py (`1b3c9c330`,
   oracle sharding) is gated entirely on `--oracles` (functions only called under
   `if args.oracles`) and explicitly preserves the per-shape reset.

2. **Empirical: same-inductor cross-run drift = the A/A floor.** New-baseline vs
   prior-baseline (both 244fdb379d11) per-point |%Δ|: median 1.13%, p95 8.52%.
   New-branch vs prior-branch: median 1.16%, p95 8.59%. Both match the documented
   A/A floor (per-point median 1.12%, p95 8.4%). SIGNED drift is centered at 0
   (baseline median +0.006%, branch +0.000%) → no systematic shift from the
   harness change or from contention.

## (d) Adversarial checks

- **Cache isolation CONFIRMED**: baseline `/tmp/tmp.t3gb9pluRa` ≠ branch
  `/tmp/tmp.hDYdEN7BVr`; both freshly `mktemp -d`'d per arm. No cross-arm codegen
  leak (which would zero the delta — it did not; branch SOL 42% vs baseline 36%).
- **Swap integrity CONFIRMED**: baseline arm = 28 inductor files modified vs
  tmp_work (baseline tree active); branch arm = 0 (clean tmp_work). Matches prior
  run exactly.
- **Hand-checked 3 models term-by-term** (per-kernel base/branch us from the new
  arms × occurrence sidecar):
  - timm/infer/adv_inception_v3: NEW +26.40% vs PRIOR +26.64% (Δ -0.23pp). Big
    reduction wins reproduce (b71f0d447f2e 216→113us, b0a24575ebc9 349→145us).
  - hf/infer/AllenaiLongformerBase: NEW -9.73% vs PRIOR -9.40% (Δ -0.34pp). The
    documented amax_sum regression reproduces (528a3c274a41 636→786us ×12).
  - hf/infer/BertForMaskedLM: NEW +5.54% vs PRIOR +6.17% (Δ -0.63pp). 93ab7097e9e5
    628→213us. All within band.
- **Genai excluded** in the genai-excl cut (8 genai/static models). Both cuts
  reported; the headline holds in either.
- **A/A floor re-confirmed**: per-kernel cross-run drift reproduces the ±0.82%
  per-model / 8.4% per-point band; per-model improvement deltas median |Δ| 0.23%.

### Caveat (honest)
- Per-MODEL improvement drift p95 = 1.64% (23/158 models exceed ±0.82%), higher
  than a same-session A/A's 0.82%. Expected: this is a cross-RUN diff (different
  day, light concurrent CPU-accounting load) and combines noise from BOTH arms.
  The movers are all tiny/fast train models (tts_angular, lennard_jones, GPTJ
  near-zero) where small absolute jitter is large %. The MEDIAN/MEAN headline is
  unaffected (Δ ≤ 0.05pp).
- A foreign `model_graph_accounting.py` (≤51GB GPU mem) ran concurrently during
  the baseline arm. It is a pure-arithmetic accounting tool (reads pre-computed
  `--timings` JSON; NO `do_bench` / timing loop), so it cannot contaminate timed
  replays; the bench lock + min-of-N + the ~0 signed drift confirm no bias.

## (e) Root cause if not reproduced
N/A — it reproduced.

## (f) Artifacts (durable, results/perf_ab/revalidate_2026-06-18/)
- `baseline_kernels.json`, `branch_kernels.json` — the two fresh A/B arms (4977 pts each)
- `baseline_run.log`, `branch_run.log`, `*_timing.txt`, `*_cache_dir.txt` — run provenance
- `reconcile.py` + `reconcile.json` — Deliverable 1 recompute + per-model drift vs prior
- `kernel_drift.py` + `kernel_drift.json` — per-kernel cross-run A/A drift (both arms)
- `handcheck.py` + `handcheck.json` — 3-model term-by-term audit
- `REVALIDATION_SUMMARY.md` — this file
