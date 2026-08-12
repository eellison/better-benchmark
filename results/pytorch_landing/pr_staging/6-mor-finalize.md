# PR 6 — Triton finalize-sum kernel for multi-output split reductions (MOR) — U30

**Unit:** U30, carved from mega-commit `97385fb3273`.
**Flag:** `config.triton_finalize_sum` (`TORCHINDUCTOR_TRITON_FINALIZE_SUM`,
default on) — **added by this carve** as a kill switch; the original mega
codegen was unconditional.
**Status:** MERGE-READY, two commits — `pr/mor-finalize-clean`.

## Branch & how made mergeable
`pr/mor-finalize-clean` (rooted on `5e2ab3055de`):
- `f8edc3e16a2` — **feature**, carved from mega `97385fb3273`: the complete
  `torch/_inductor/codegen/simd.py` diff (+155/−21; the only file the unit
  touches): `nsplit_expr` refactor, the `use_triton_finalize` branch in the
  `saved_partial_accumulate` finalize path, and the new
  `_emit_triton_finalize_sum` method (~120 lines).
- `8d664ddc25e` — **kill switch, separate and clearly labeled as not part of
  the original feature**: adds `config.triton_finalize_sum` (default on) and
  threads it into the `use_triton_finalize` condition, so reviewers/users can
  restore the ATen `.sum(dim=0)` fallback without a source change.

**Entanglement check (per CORE_REVERT_RESULTS.json U30 scope):** the revert
scope was simd.py-only ("forced use_triton_finalize=False … reverts the
+176-line branch"); its inverse — this carve — needs **no wrapper/ir/triton.py
hunks**. The mega's `ir.py` `_sibling_reduction_fusion` hunk is gated by a
config that is `False` at HEAD (dead code) and is NOT carried. All symbols the
codegen uses (`last_power_of_2`, `DeviceProperties`, `IndentedBuffer`,
`textwrap`, `triton_heuristics.fixed_config`, `FixedGrid`/`extra_launcher_args`
launcher support, `sizevars.optimization_hint`, `wrapper.define_kernel`,
`write_get_raw_stream`, `wrapper.allocated`) exist at base `5e2ab3055de`
(checked individually). The carve is faithful: branch simd.py differs from the
mega's simd.py ONLY by the added `config.triton_finalize_sum and` line.

## Verification (2026-07-15, B200)
- **Apply-clean:** `git apply --check` + `git am` of `patches/6-mor-finalize.patch`
  onto a pristine `5e2ab3055de` worktree — clean; 0 conflict markers.
- **AST-parse:** both touched files parse.
- **Shadow-import:** PYTHONPATH-shadow over the prebuilt HEAD `.so`;
  `torch._inductor.codegen.simd` imports OK; `config.triton_finalize_sum == True`
  (absent on the base arm).
- **Compile-smoke:** `repros/canonical/sum_sum_sum_260a107eaf32` (the repro
  CORE_REVERT used) `torch.compile`s to completion — **GOOD**. The codegen
  **FIRES**: branch generated code contains **4 `triton_mor_finalize_sum`
  kernel defs (base: 0)** and aggregate `@triton.jit` count **10 vs 6** — the
  documented njit 3→5-per-graph relation (this smoke compiles 2 graphs).
- **Kill switch:** `TORCHINDUCTOR_TRITON_FINALIZE_SUM=0` on the branch arm
  restores base codegen exactly (0 markers, njit 6).
- **Numerics:** compiled-vs-eager and vs an fp64 reference on all 6 outputs.
  The finalize-summed outputs are *closer* to fp64 than eager (compiled_err
  6.1e-5 vs eager_err 7.6e-5 — the fp32 workspace + tl.sum tree beats eager's
  bf16 accumulation order). One output (out[5]) shows a large compiled-vs-eager
  drift (2.5e3 on ~1e3-magnitude sums of squares) that is **pre-existing at
  base** — the base arm produces the identical numbers (split-reduction
  reassociation of a bf16 dot-square chain), not introduced by this PR;
  flagged per the numerics-nonblocking policy. **Net: imports + compiles the
  representative repro; the new kernel's own outputs are numerics-valid
  (better-than-eager); full CI not run.**

## Perf verification (2026-07-15, B200): REPRODUCES
A/B PYTHONPATH-shadow base `5e2ab3055de` vs branch tip `8d664ddc25e`, fresh
inductor cache per arm, bench_parallel locked path (`--no-cd`, clocks 1852MHz),
6 of the 52 affected sum_sum_sum repros, `--all-shapes` = 13 shape points:

| metric | value |
|---|---|
| geomean speedup (as measured) | **1.173x** |
| expected | ~1.14x (CORE_REVERT 1.135x, u30_e2e re-run 1.140x) |
| range | 12/13 points 1.04x–1.44x faster |
| within-session noise (branch run1 vs run2) | 1.0132 |

Top movers: `sum_sum_sum_11d45d703ba6`/XLNet 61.3→42.6us (**1.44x** — matches
CORE_REVERT's documented best on the same repro, 1.46x),
`260a107eaf32`/Bart 69.1→48.8us (1.42x), /Electra 63.5→46.8us (1.36x),
`2261b2f5694a`/MobileBert 64.8→49.0us (1.32x).

**The one negative reading does not survive re-measurement:**
`sum_sum_sum_ecce309d13e3`/LayoutLM read 0.72x in the batch A/B, but the point
is autotune-bistable — the branch's own two batch runs differ 1.19x
(462.0 vs 386.9us) and the BASE arm itself swings 330.6 → 560.7/560.8us across
fresh-cache runs (1.7x self-variance on a 15-kernel graph). Isolated sequential
re-bench (fresh caches, 2 runs per arm): base 560.7/560.8 vs branch
543.6/545.6us → branch **faster** (1.03x). Geomean excluding the bistable
point: 1.223x; with the isolated-corrected value: 1.207x. No reproducible
regression. Raw data: `perf_verify/RESULTS_5_6.json` (this dir).

## Summary
When a multi-output split reduction saves partial sums to a workspace, the
finalize step previously fell back to ATen's generic `.sum(dim=0)` over the
`[nsplit, rnumel]` workspace view. This PR emits a dedicated 2D Triton finalize
kernel instead (`triton_mor_finalize_sum_*`): each program loads a
`[BLOCK_T, BLOCK_C]` sub-tile with coalesced column access, partial-sums over
the tile axis, and `atomic_add`s into a zero-initialized output. Applies to
`reduction_type == "sum"` on CUDA python-wrapper builds only; everything else
keeps the ATen fallback.

## Measured impact (kernel + e2e, honestly)
- **1.14x kernel geomean over the 52 affected sum_sum_sum repros** (measured
  twice at HEAD: 1.135x / 1.140x; this carve vs clean base: 1.17x on the 6-repro
  subset), 48/52 help, **0 regress**, best 1.46x.
- **e2e: +0.25pp corpus geomean (n=158, genai-excl)** — below the ±0.82pp
  per-model floor in aggregate — but **+0.95pp geomean on the 29 touched
  transformer/timm TRAIN models** (median +1.0pp, up to +2.4pp Electra),
  **0 regressions**; uniform sign across 28 independent models is signal.
  Land as a kernel-codegen improvement with a modest, positive,
  TRAIN-concentrated e2e contribution — not on a corpus-wide e2e claim.

## Mechanism
ATen's generic `.sum(dim=0)` on a `[nsplit, rnumel]` strided view is a
column-strided reduce with poor coalescing at these shapes (nsplit 8–128,
rnumel 1k–8k slivers). The generated 2D kernel tiles both axes so each warp
reads contiguous columns, and the atomic_add finalize over the small output is
cheap. Win is uniform and structural (the same kernel replaces the same ATen
call on all 52 repros).

## Test plan
sum_sum_sum canonical family (BN-backward / sibling-reduction tails of XLNet,
MobileBert, Bart, DebertaV2, MegatronBert, beit, mobilevit TRAIN graphs);
numerics-gated vs eager (finalize outputs also validated against fp64);
kill-switch A/B (`TORCHINDUCTOR_TRITON_FINALIZE_SUM=0`) confirms codegen parity
with base. Marker test: `triton_mor_finalize_sum` present iff flag on, CUDA,
python wrapper, sum reduction.

written with claude code
