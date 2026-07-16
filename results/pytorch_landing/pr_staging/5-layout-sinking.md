# PR 5 — Layout-transform store sinking (channel shuffle) — U10

> **STATUS: DEPRIORITIZED (2026-07-16, maintainer call).** Kernel win is real
> (1.38x, 18/18, 0 regress) but e2e impact is ShuffleNet-local and small — not
> worth standalone upstream effort. Branch + evidence kept; revisit only if
> layout-transform patterns show up in models that matter.

**Unit:** U10, carved from mega-commit `97385fb3273`.
**Flag:** `config.layout_transform_store_sinking`
(`TORCHINDUCTOR_LAYOUT_TRANSFORM_STORE_SINKING`, default on).
**Status:** MERGE-READY, single standalone commit — `pr/layout-sinking-clean`.

## Branch & how made mergeable
`pr/layout-sinking-clean` (rooted on `5e2ab3055de`):
- `f56267540e4` — **ONE commit**, carved from mega `97385fb3273`: the new file
  `torch/_inductor/fx_passes/layout_transform_store_sinking.py` (260 lines,
  byte-identical to the mega's version), its registration block in
  `fx_passes/post_grad.py` (after `partitioned_scatter_optimization`, mirroring
  the mega's placement), and the config flag in `config.py`.

No prerequisite commit needed: the pass imports only `logging`, `typing`,
`torch`, `torch.fx`, and `torch._dynamo.utils.counters` — all present at base.
The registration uses only `GraphTransformObserver` + `config`, both at base.

## Verification (2026-07-15, B200)
- **Apply-clean:** `git apply --check` + `git am` of `patches/5-layout-sinking.patch`
  onto a pristine `5e2ab3055de` worktree — clean; 0 conflict markers.
- **AST-parse:** all 3 touched files parse.
- **Shadow-import:** PYTHONPATH-shadow worktree over the prebuilt HEAD `.so`;
  `torch._inductor.config` / `fx_passes.post_grad` import OK;
  `config.layout_transform_store_sinking == True` (absent on the base arm).
- **Compile-smoke:** `repros/canonical/pointwise_00475df23925` (shufflenet_v2
  channel-shuffle BN tail) `torch.compile`s to completion — **GOOD**. The pass
  **FIRES**: `counters['inductor']['layout_transform_store_sinking'] = 1` and the
  kernel structure changes — branch emits **2 triton kernels vs 4 on base** (the
  clone kernel is gone; the flag-ablation's documented nk 1↔3 relation, seen here
  as 2↔4 because the smoke's default shape carries an extra reduction). Second
  repro `var_mean_33d24326eae6`: counter=1 as well.
- **Flag kill-switch:** `TORCHINDUCTOR_LAYOUT_TRANSFORM_STORE_SINKING=0` on the
  branch arm restores base codegen exactly (counter 0, njit 4).
- **Numerics:** compiled-vs-eager identical between base arm and branch arm on
  both smoked repros (max_abs 9.8e-4 at bf16 scale on the pointwise repro,
  1.0e-8..2.4e-4 on the var_mean repro; same values both arms — the rewrite is
  layout-only). **Net: imports + compiles representative repros to
  numerics-valid results on B200; full CI not run.**

## Perf verification (2026-07-15, B200): REPRODUCES (exceeds)
A/B PYTHONPATH-shadow base `5e2ab3055de` vs branch tip `f56267540e4`, fresh
inductor cache per arm, bench_parallel locked path (`--no-cd`, clocks 1852MHz),
**all 14 flag-ablation-affected shufflenet_v2 repros** (`mega_ablation`
affected_U10.txt), `--all-shapes` = 18 shape points:

| metric | value |
|---|---|
| geomean speedup | **1.379x** |
| range | 1.14x – 2.37x, **18/18 faster, 0 regress** |
| within-session noise (branch run1 vs run2) | 0.9875 (≈1% floor) |

Top movers: `var_mean_var_mean_92c45aff3580` 57.1→24.1us (**2.37x**, nk 4→2),
`var_mean_var_mean_e642e2ea37a3` 34.7→19.5us (1.78x), `var_mean_33d24326eae6`
32.3→19.3us (1.67x), `pointwise_ba25986618ef` 27.4→18.0us (1.53x, nk 4→1).
Structural: n_kernels drops on 12/18 points (2→1, 4→1, 4→2, 6→4).

**Expected was ~1.20x** (the flag ablation's 1.2003, measured as the marginal
contribution *at HEAD* `daa79cd25ca` where other landed units had already sped
these kernels up). Against clean base the same pass removes more of the
remaining gap, hence 1.38x — direction and surface identical, magnitude larger.
Raw data: `perf_verify/RESULTS_5_6.json` (this dir).

## Summary
A post-grad FX pass that detects the channel-shuffle layout-transform pattern
`cat(dim=D) → view(split D into groups×C/g) → permute(swap D,D+1) →
clone(contiguous) → view(final)` and rewrites it to
`[unsqueeze(inp, D+1)] → cat(dim=D+1) → view(final)`. The cat on the unsqueezed
dim produces the same memory layout the permute+clone did, so the clone kernel
is eliminated entirely and the cat fuses with its upstream producers. ~260
lines in a new file + a config flag + a registration block.

## Measured impact (kernel + e2e, honestly)
- **1.38x geomean on its 14 affected kernels vs clean base** (1.20x measured as
  marginal-at-HEAD in the flag ablation); far above both noise floors.
- **e2e: small and ShuffleNet-local.** All 14 affected repros in the 1727-repro
  corpus are shufflenet_v2_x1_0; the affected kernels are 10–160us slivers, so
  per-model e2e impact is bounded to ShuffleNetV2's layout-transform fraction.
  Do not attach a corpus-wide pp claim; this lands as a kernel-codegen win with
  a pattern (grouped-conv channel shuffle) that generalizes beyond this corpus.

## Mechanism
The clone in a channel shuffle is a pure layout transform: every element is
read once and written once to a permuted address. Sinking the store into the
producer (via the unsqueeze+cat rewrite) deletes an entire memory round-trip
over the activation tensor — on memory-bound B200 kernels that is the whole
kernel's cost. The structural tell is the kernel-count drop.

## Test plan
The 14 canonical shufflenet_v2 repros (`pointwise_00475df23925` family +
`var_mean_*`/`var_mean_var_mean_*` shuffle tails); numerics-gated vs eager;
flag-off A/B to confirm codegen parity with base. Pattern unit test: the
rewrite requires single-user chain cat→view→permute→clone→view,
contiguous-format clone, static groups/C-per-group matching the cat inputs —
each guard has a negative case.

written with claude code
