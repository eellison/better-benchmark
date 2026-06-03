# Benchmark Infrastructure Validation & Full Sweep

## Status: COMPLETE

## Context
- Local optimization changes stashed (stash@{0})
- Checked out PR #184905 (pytorch/pytorch) as clean baseline
- Goal: validate infra, run full parallel sweep, then test with tuning knobs

## Tasks

### 1. Rebuild PyTorch on PR #184905
- [x] Stash local optimizations (stash@{0})
- [x] Checkout pr-184905 branch
- [x] Rebuild complete: torch 2.13.0a0+git111efb8 (PR #184905, commit 111efb8)
  - Fixed duplicate enum_tag.h include guard issue in build
  - 2x NVIDIA B200 GPUs detected

### 2. Validate benchmark infra works on PR branch
- [x] Confirm repro_harness.py runs correctly (CUDAGraph timing)
- [x] Confirm bench_parallel.py runs with multi-GPU parallelism (2x B200)
- [x] Quick smoke test: 5/5 repros passed, timing values sane
  - pointwise_99db949c0182: 34.6us compiled, gap=0.99x (at SOL)
  - sum_558522245da0: gap=1.53x
  - var_mean_dc2d388af48f: gap=1.93x
  - var_mean_mean_626fb0807512: gap=1.64x
  - pointwise_2eae25328292: gap=4.14x
  - Output: /tmp/smoke_test.json

### 3. Full baseline sweep (parallel, multi-compilation per GPU)
- [x] Run full corpus sweep on PR #184905 (clean, no local changes)
- [x] Saved as sweep_pr184905_baseline.json
  - 1482/1482 repros passed (0 failures)
  - Total time: ~12 minutes (--workers-per-gpu 4, 2x B200)
  - Default compilation: median gap 1.74x, mean 2.53x, 22.0% at SOL
  - Coordinate descent: median gap 1.60x, mean 2.15x, 25.5% at SOL

### 4. Sweep with tuning knobs enabled
- [x] coordinate_descent_tuning=True only (included in baseline sweep)
  - Median gap: 1.60x (vs 1.74x default), 25.5% at SOL (vs 22.0%)
- [x] combo_kernels=True (full corpus, re-run 2026-06-03)
  - 1482/1482 passed (0 failures) in 464s
  - Median gap: 1.56x, Mean gap: 2.11x, 27% at SOL
  - Output: sweep_pr184905_all_knobs.json
- [x] multi_kernel=1 attempted but NOT viable
  - 1298/1482 repros FAIL (87.6% failure rate)
  - Primary error: TypeError: 'NoneType' object does not support the context manager protocol
  - multi_kernel is broken on this branch for this corpus

### 5. Compare results: baseline vs combo_kernels
- [x] Full comparison (1482 common repros)
  - DEFAULT timing geomean: 1.0024x (combo_kernels 0.24% faster on average)
  - CD timing geomean: 1.0080x (combo_kernels 0.80% faster on average)
  - DEFAULT: 160 improved >5%, 100 regressed >5%, 1222 same
  - CD: 181 improved >5%, 113 regressed >5%, 1188 same
  - Top win: pointwise_c14f03aac63b 3.16x faster (38.6us -> 12.2us)
  - Worst regression: sum_sum_sum_f0377fc40fe2 6.9x slower (60.3us -> 417.5us)
  - Conclusion: combo_kernels is essentially neutral on average, with targeted
    wins on pointwise fusions and some regressions on sum/reduction patterns
  - Output: sweep_pr184905_comparison.txt

## Summary Table

| Config | Pass Rate | Median Gap | Mean Gap | At SOL (<=1.1x) |
|--------|-----------|------------|----------|-----------------|
| Default (no tuning) | 1482/1482 (100%) | 1.74x | 2.53x | 22.0% |
| Coordinate Descent only | 1482/1482 (100%) | 1.60x | 2.15x | 25.5% |
| combo_kernels=True | 1482/1482 (100%) | 1.56x | 2.11x | 27.0% |
| multi_kernel=1 | 184/1482 (12%) | N/A | N/A | N/A (broken) |

## Key Files
- Baseline sweep: `sweep_pr184905_baseline.json`
- All-knobs sweep (combo_kernels): `sweep_pr184905_all_knobs.json`
- Comparison report: `sweep_pr184905_comparison.txt`
- Failed multi_kernel attempt: `sweep_pr184905_all_knobs_v2.json` (184/1482 only)

## Notes
- PR #184905 branch: `pr-184905` (commit 111efb8724a)
- Stashed optimizations: `stash@{0}` on branch `reduction-transpose-only`
- Build fix: added include guard to `build/torch/headeronly/core/enum_tag.h` and `build/aten/src/ATen/core/enum_tag.h` to fix duplicate enum definition
- Added `--combo-kernels` and `--multi-kernel` CLI flags to bench_parallel.py
