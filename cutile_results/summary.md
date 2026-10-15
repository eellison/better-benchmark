# cuTile vs Triton — Compiler Comparison Results

## What this is

The `better-benchmark` corpus ships each canonical kernel pattern with a
Triton oracle (a hand-written reference kernel that serves as the
optimization target for `torch.compile`'s output). This experiment adds a
**cuTile port for every one of those patterns** and benchmarks both against
`torch.compile` with the same harness, so the two GPU DSLs can be compared
head-to-head on the same workload.

The goal is a **compiler-vs-compiler** comparison: Triton's compiler vs
NVIDIA's cuTile compiler. Both ports do the same computation with the same
kernel structure — differences in wall time reflect codegen quality, not
port quality.

## Corpus coverage

* **1727 canonical kernel patterns** total in the corpus.
* **1717 real cuTile ports** (99.4%). Each has at least one `@ct.kernel`
  doing substantive work; the numerics gate passed against the eager
  `Repro`. Ports live under `repros_cutile/canonical/<name>/oracle.py`.
* **10 remaining stubs** — patterns where after multiple rewrite attempts
  cuTile numerics diverged past the 1% tolerance (e.g. PyHPC Thomas
  solver, a 30-kernel serial stencil).

## Fairness discipline

The point of the comparison is compiler codegen quality, not port skill.
Two rounds of audits were applied to eliminate unfair advantages / handicaps:

1. **240 cuTile ports had torch reductions / erf / softmax in
   `oracle_forward`** while the corresponding Triton oracle kept those
   inside its kernel body — these were rewritten to compute in-kernel
   via `ct.sum` / `ct.exp` / an Abramowitz-Stegun erf polynomial.

2. **150 top-loss cuTile ports had code-quality antipatterns** — mostly
   unnecessary `.contiguous()` calls on views that were already
   contiguous, `torch.zeros(padded); pad[:H,:W] = src` style pad-copies
   that could have been expressed with `padding_mode=ct.PaddingMode.ZERO`,
   and tiny BLOCK sizes (`BLOCK=1` or `4`) when Triton used
   `BLOCK=1024`. These were rewritten to remove the antipattern while
   preserving the Triton kernel structure exactly.

After these fixes cuTile mirrors Triton in **number of kernels, number of
`ct.launch` calls, and BLOCK sizes** wherever the numerics gate allows.

## Setup

* Hardware: 2× NVIDIA B200
* Harness: `scripts/bench_parallel.py --oracles`
* Timing: `--n-warmup 10 --n-rep 40 --rounds 5`, min-of-N across rounds,
  each run inside an exclusive GPU flock so per-GPU timing is isolated
  from cross-process interference.
* Measurement path: `torch.compile` compiles the eager `Repro` for the
  same shape, then CUDA-graph-captures both compile-output and the
  oracle. Times are captured graph-replay times.
* Two data sets were merged for coverage:
  * **`triton_787dirs_all_shapes.json` + `cutile_787dirs_all_shapes_post_fix.json`**:
    every shape point in the first-wave 787-dir subset. This gives
    per-shape resolution for the well-covered subset.
  * **`triton_1717dirs_first_shape.json` + `cutile_1717dirs_first_shape.json`**:
    one representative shape per canonical dir across the full 1717-dir
    corpus. This gives broad coverage of the long tail.
  * **`triton_merged.json` + `cutile_merged.json`**: the union.

## Headline (merged coverage — 2077 comparable points)

| Metric                              | Value      |
|-------------------------------------|-----------:|
| Total (dir, shape) points measured  | 3089       |
| Valid Triton timings                | 2831       |
| Valid cuTile timings                | 2195       |
| **Comparable (both valid)**         | **2077**   |
| **Geomean cuTile / Triton**         | **1.71×**  |
| **Median cuTile / Triton**          | **1.11×**  |
| cuTile faster than Triton           | 225 (11%)  |
| Tied within noise                   | 52 (2.5%)  |
| Triton faster than cuTile           | 1800 (87%) |

Read: on the median kernel, cuTile is **11% slower** than Triton. On the
geomean, cuTile is **71% slower** — the gap between median and geomean is
the long tail of hard cases (see below).

## `torch.compile` floor comparison

Each oracle is graded against `torch.compile`'s output for the same
`Repro`:

| Status                      | Triton | cuTile |
|-----------------------------|-------:|-------:|
| GOOD (oracle beats compile) |    493 |     67 |
| AT_FLOOR (matches compile)  |   1737 |    745 |
| BAD_ORACLE (slower)         |    601 |   1383 |
| NUMERICS_WORSE_THAN_COMPILED|    138 |    155 |

Triton lands at or above the `torch.compile` floor **79%** of the time.
cuTile lands there **33%** of the time.

## Top 10 kernels where cuTile beats Triton

| Kernel                                    | Shape hash | Triton (µs) | cuTile (µs) | Ratio   |
|-------------------------------------------|-----------:|------------:|------------:|--------:|
| pointwise_f662d0fd23e3 (GELU+dropout)     | c78a05f8   |      138.91 |       50.88 | 0.37×   |
| pointwise_301cf5a13527 (fused pointwise)  | 1a8eaeba   |       23.30 |       16.86 | 0.72×   |
| sum_1bd9fae13cde (vocab column sum)       | ceaa9c1c   |     1235.81 |      925.50 | 0.75×   |
| pointwise_e87d6ebc9ded (bias + activation)| 226fbbfa   |       17.34 |       14.56 | 0.84×   |
| argmax_amax_sum_a0354c147dd8              | 91cac6df   |        8.99 |        7.58 | 0.84×   |
| pointwise_e87d6ebc9ded                    | c23ba4e7   |       11.14 |        9.60 | 0.86×   |
| pointwise_301cf5a13527                    | b8160d07   |       11.23 |        9.70 | 0.86×   |
| pointwise_a7159158fa8e                    | 200ac53a   |       12.90 |       11.20 | 0.87×   |
| argmax_amax_sum_52310a7c5196              | d21e6ec7   |        9.06 |        7.94 | 0.88×   |
| pointwise_301cf5a13527                    | ad7b2a2c   |       11.46 |       10.08 | 0.88×   |

Simple pointwise and single-axis reductions.

## Top 10 kernels where cuTile loses hardest

| Kernel                                    | Shape hash | Triton (µs) | cuTile (µs) | Ratio    |
|-------------------------------------------|-----------:|------------:|------------:|---------:|
| sum_sum_sum_127fc8edd5da                  | 953b9872   |      135.10 |    51313.76 | **380×** |
| sum_sum_sum_d0b0dff4e37d                  | 7c1570d5   |      155.30 |    52487.17 | 338×     |
| sum_sum_sum_65c3d9f36fd9 (LN backward)    | e3ecf7e0   |       43.10 |     9959.39 | 231×     |
| sum_sum_sum_cb474de4ede0                  | 4e4a9284   |      141.15 |    15839.10 | 112×     |
| pointwise_b43af69d4124 (Demucs slice+mask)| 3d3e6f4d   |       30.62 |     3057.54 | 100×     |
| sum_sum_sum_c0d30ef6f9c5 (RepVGG BN-bwd)  | ac965caa   |       57.06 |     5160.96 | 90×      |
| sum_sum_sum_4ce9013c6e0d                  | 409a14a3   |       24.38 |     2044.99 | 84×      |
| sum_sum_d283dea24dab (BN backward)        | d56aace4   |       52.93 |     3887.97 | 73×      |
| var_mean_c003cf6c87f4 (Channels-last LN)  | 8b7d5a32   |       68.35 |     4513.66 | 66×      |
| sum_sum_sum_0e08b9d50286 (embedding grad) | 05376402   |       69.31 |     4441.25 | 64×      |

## Analysis

### 1. cuTile is competitive on simple pointwise / single-axis reductions

The median case (1.11× slower) and the top wins both suggest cuTile's
compiler produces near-Triton-quality code for straightforward pointwise
kernels and single-axis reductions. Both compilers seem to lower these
patterns effectively; the small deltas are probably launch overhead and
register-pressure differences.

### 2. The long tail is not code quality — it's cuTile compiler weakness on specific patterns

Every top-loss kernel was audited (and re-audited) for antipatterns.
None of the 20 worst losses have unfair torch offloading, `.contiguous()`
copies, or tiny BLOCK sizes. The cuTile ports look structurally identical
to the Triton ports — but they run 30-400× slower. This is a genuine
codegen gap.

Concrete failure modes observed:

* **Cooperative split-K reductions with multiple outputs** (e.g.
  `sum_sum_sum_127fc8edd5da`, `sum_sum_sum_65c3d9f36fd9`). Triton's split-K
  reduces to launching one kernel per split with per-CTA reductions and
  then a finalizer that sums per-CTA partials. cuTile ports that mirror
  this structure run drastically slower — likely because cuTile's
  cross-CTA reduction primitives don't map to the same warp-shuffle
  strategy Triton uses.

* **Channels-last non-power-of-2 reductions** (e.g. `var_mean_c003cf6c87f4`,
  `var_mean_e98d6d833b6e`). Triton's `mask`-based tail handling produces
  compact PTX; cuTile's `PaddingMode.ZERO` on non-pow-2 loads seems to
  block coalescing or vectorization on the tail.

* **Very-small-work kernels launched over huge grids** (e.g.
  `pointwise_b43af69d4124`, which processes 4-element blocks 5.9M times).
  Triton compiles these to trivially small kernels with high grid
  occupancy. cuTile's overhead per program appears to dominate here.

* **Multi-input residual chains** (`sum_sum_sum_cb474de4ede0`,
  `sum_sum_sum_4ce9013c6e0d`). Triton fuses ~20 pointwise ops that write
  to shared row-reductions in one kernel; cuTile's fused version of the
  same structure runs 30-100× slower even though the kernel body is
  identical shape to Triton's.

### 3. cuTile beats `torch.compile` far less than Triton does

Triton oracles beat/match `torch.compile` on **79%** of kernels; cuTile
oracles do the same on **33%**. Because `torch.compile` itself produces
Triton (via TorchInductor), Triton is playing at home. But 33% is much
lower than one would expect from "another optimizing GPU compiler on the
same problems" — suggesting cuTile's compiler still has meaningful room
to close on optimization patterns TorchInductor already generates well.

### 4. Fairness controls affected the numbers materially

Before the fairness+perf-fix waves, the same 787-dir subset showed
**geomean 2.10× / median 1.17×**. After fixes:
**geomean 1.71× / median 1.11×**. The gap tightened by ~40% in geomean
(a lot of it was avoidable `.contiguous()` copies and torch offloads).
But it did **not** close on the top losses — those really are cuTile
compiler codegen issues.

## What's next

* Concrete cuTile compiler issues worth investigating:
  * Cooperative multi-output split-K reductions
  * `PaddingMode.ZERO` tail handling on non-pow-2 loads
  * Small-kernel launch overhead at very-high grid counts
* The 10 remaining stubs are patterns with numerics-boundary issues
  (bf16 accumulation ordering, `prims.fma` vs separate mul+add) — could
  be revisited if `torch.allclose(atol=1e-2, rtol=1e-2)` is relaxed for
  the specific bit-level-sensitive patterns.

## Files in this directory

| File                                            | Description |
|-------------------------------------------------|-------------|
| `summary.md`                                    | This file.  |
| `triton_merged.json`                            | Triton per-shape timings, merged 787-all-shapes + 1717-first-shape. |
| `cutile_merged.json`                            | cuTile per-shape timings, merged; **post-fairness+perf fixes**. |
| `comparison_merged.json`                        | Per-row comparison + summary stats. |
| `comparison_merged.csv`                         | Same, CSV form (columns: dir, shape_hash, triton_us, cutile_us, cutile_over_triton, statuses). |
| `triton_787dirs_all_shapes.json`                | Triton on original 787 dirs × all shapes. |
| `cutile_787dirs_all_shapes_post_fix.json`       | cuTile on same, post-fix. |
| `triton_1717dirs_first_shape.json`              | Triton on full 1717 dirs × 1 shape. |
| `cutile_1717dirs_first_shape.json`              | cuTile on same. |
| `comparison_pre_fix_787.json`                   | For reference — the pre-fix comparison at geomean 2.10×. Included so the fairness delta is auditable. |
