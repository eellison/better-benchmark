# Longformer band-materialization fix — IMPLEMENTATION ATTEMPT (real code + measured)

**Target:** `repros/canonical/amax_sum_528a3c274a41` point `79c25467` (Longformer sliding-window attention band). Baseline compile **786us** / oracle **319us** = 2.46x. Proven gather ceiling = 296us.

**Status: PARTIALLY CLOSED — load-bearing core BUILT & proven bit-exact; blocked on ONE precisely-pinned edge case. Gap NOT closed on the target (still 786us). No regression anywhere.**

This converts the prior "multi-week, handoff" estimate into working code + a measured boundary: the hard core (collapse a scatter-chain into one gather load) is implemented and demonstrated bit-exact on synthetics (16 loads → 1); the remaining blocker is a single source-pattern (`constant_pad_nd` conditional load) that the unifier currently (correctly) bails on, plus readback view-threading. Realistic remaining effort: **1–2 weeks**, now de-risked.

---

## Isolation (used + removed)
- `git worktree add /tmp/scratch_space/pytorch-longformer-impl HEAD` off `/tmp/pytorch-work`.
- The worktree lacks built C++ libs, so symlinked `_C*.so`, `lib/`, `bin/`, `version.py` from the shared tree; PYTHONPATH-shadowed.
- Verified before every bench: `torch._inductor.lowering` resolves to the worktree. GPU lock ON; `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`; fresh `torch._dynamo.reset()` per shape (harness 4ca6d532b present).
- All edits in the worktree; **no tracked better_benchmark files modified**; outputs under gitignored `results/longformer_amax_sum/IMPL/`.

## The patch — `scatter_chain_fold.patch` (torch/_inductor/{lowering,config,graph}.py)
Lever A at the `slice_scatter`/`select_scatter` **lowering** (not an FX pass):
1. **Structured scatter chain.** A side table maps each scatter `Pointwise` to a flat list of `_ScatterWrite(mask_fn, src_loader)`. When a scatter's base is a constant-`full()` chain, APPEND a write instead of nesting another `where` over the previous loader.
2. **View-readback threading** (`_get_chain`): threads the chain through pure-reindex views (`SliceView`/`squeeze` done) via `BaseView.make_reindexer()` (`view_idx → base_idx`), so the `select(band)…scatter…select_scatter(band)` readback idiom keeps the chain alive.
3. **Index unification** (`_build_unified_inner` — the load-bearing piece): if every write's source is a pure single-buffer affine load, collapse them to ONE `ops.load` whose index is `where`-selected per region (last-writer-wins), clamped into bounds. This is the gather. If any source isn't a pure load → **bail, keep the original nested lowering** (so the fold can never regress a non-unifiable chain).
4. `GraphLowering.run` clears the side table per compile (id-keyed, no cross-compile aliasing).

Flag `fold_scatter_chain_loader` (`TORCHINDUCTOR_FOLD_SCATTER_CHAIN`, default on; safe because it only changes codegen when it strictly collapses loads).

## Measured results (B200, CUDAGraph bench via oracle_harness)
| | compile_us | band buffers (513) | numerics |
|---|---|---|---|
| baseline (unpatched) | **786** | 6 | bit-exact |
| patched (fold on) | **786** | 6 | **bit-exact (max_diff=0.00e+00)** |

**Perf-inert on the target** (and Longformer-train sibling: 863→862). The 100MB band is NOT removed. The fold builds the chain but **index-unification bails** on Longformer (see below), so codegen is unchanged.

## Core mechanism is PROVEN (the part prior agents called multi-week)
On a direct disjoint scatter-chain-over-`full` (`unify_direct_test.py`): `tl.load` count **16 → 3**, **1** fused kernel, fold fires 14×. Bit-exact. Overlapping last-writer test (`unify_exact_correctness_test.py`): **max_diff=0.0, torch.equal=True** — the `where`-fold-first→last correctly resolves last-writer-wins even when a later write partially overwrites an earlier full write.

## Correctness bug found & fixed (validates the design-doc clamp trap)
Fold-on on demucs (`sum_1e8a518eb72e`, a lone `slice_scatter` over `full`) → **CUDA illegal memory access**. Cause: the unified `ops.load` is evaluated unconditionally (then `where`-masked); at uncovered positions the selected index is out of bounds → IMA. **Fix:** clamp the index into `[0, numel)` before `indirect_indexing` (mirrors the proven gather rewrite's `.clamp()`); the final `where` still discards it. After fix: PASS, no IMA, codegen identical to baseline.

## Why inert on the target — root cause PINNED (runtime evidence)
- **Chain self-breaks at realization.** `realize_interleave_trace.py`: `FOLD nwrites=1 → FOLD nwrites=2 → REALIZE (96,4,256,513) in_chain_table=True`. The 2-write chain's opcount ≈40 (each write source is an ~18-op pad+permute+view chain), trips `has_large_inner_fn`(>30), realizes; later scatters then read the realized buffer, not the chain.
- **Unification — the only thing that drops opcount — bails** because Longformer write sources are **conditional loads**, not pure loads. `probe_src_pad_pattern.py` shows the source op tree: `[ge, lt, masked{load:arg1_1}, ge, lt, masked{constant}, where]` = `constant_pad_nd` (pad the bmm by one row): `where(pad_bounds, masked_load(arg1_1), pad_const)`. The pure-load probe correctly rejects it.
- **Pure laziness is not the fix** (re-confirmed here): all-thresholds=∞ → 0 band buffers but ~4× SLOWER (8468 vs 2078us wall), because the lazy tower is 5057 ops / 273 loads per element with no flat index. The win **requires** the unified single load.

## Remaining hard part (precise)
Extend index-unification to the `constant_pad_nd` conditional source `where(pad_bounds, masked_load(buf, idx), pad_const)`:
1. A **symbolic-value probe** that tracks which produced value is the load result through the `where`/`masked` tree, extracting `(buf, idx_expr, pad_bounds, pad_const)` — not just flagging ops.
2. In the unified emit, fold `pad_bounds` into the per-write coverage mask and substitute `pad_const` out-of-pad (kept **distinct** from the band base fill, the −∞ invalid-key fill, and the −3.39e38 key-bias — the design-doc traps).
3. Keep the clamp.
Plus: thread the chain through `ReinterpretView`/`PermuteView`/`View` readbacks (SliceView/squeeze already done) so the full band assembles as one chain before the realize decision. (Once unification drops the opcount, the chain stops self-realizing, threading completes it, and the existing-codegen gather — proven 296us — should fall out.)

## Remaining-effort estimate
**1–2 weeks**, now de-risked: the load-bearing unifier is built and bit-exact (incl. overlap + IMA clamp). Remaining = the pad conditional-source extractor + multi-fill recompose (~1 wk, correctness-sensitive) + view-readback threading (~2–4 days). Materially less than the FX-pattern path (no `diagonal_skew_elimination` ordering fight, no all-consumers-rewritability proof) — it stops defeating the gather the lazy IR already is, rather than reconstructing one.

## Files
- `scatter_chain_fold.patch` — the patch (unified diff vs torch/_inductor).
- `unify_direct_test.py`, `unify_exact_correctness_test.py` — the proven core (16→1 loads; overlap bit-exact).
- `realize_interleave_trace.py` — fold-vs-realize interleave (the 2-write self-break).
- `probe_src_pad_pattern.py` — the `constant_pad_nd` conditional-source op tree (the blocker).
- `probe_chain_opcounts.py` — per-scatter opcounts.
- `triton_fold_{on,off}.py` — dumped Triton (6 band buffers both).
- `baseline_bench_stdout.txt` — baseline harness output.
- `RESULTS.json` — machine-readable.
