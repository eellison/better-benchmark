# Dynamic vs static kernels across eval points — first generalization curves

**Date:** 2026-08-12 · **Hardware:** B200 · **Branch:** dynamic-shapes-capture (PR #80 head)

First cut of the "generalization curve": for a dynamic family, time the ONE
general dynamic artifact at several binding points and compare each point
against a per-shape STATIC specialized compile at the same binding. Both arms
use the identical methodology (repro self-bench: GPU lock, CUDAGraph
capture+replay, do_bench min; dynamic = mark_dynamic with a
two-distinct-shape pre-warm so the GENERAL kernel is the one timed).

Families are live captures from `scripts/exercise_dynamic_shapes_live.py`
(scratch root; the checked-in corpus is not involved). Reproduce with:

```bash
python scripts/exercise_dynamic_shapes_live.py --output-root /tmp/scratch_space/exercise_live
cd /tmp/scratch_space/exercise_live/repros/canonical
python <dir>/repro.py             --bind ... --bind ... --output static.json
python <dir>/repro.py --dynamic   --bind ... --bind ... --output dynamic.json
```

Raw per-run JSON: `dynamic_vs_static_eval_points_raw_2026-08-12.json`.

## Results

| family (live capture) | binding | static µs | dynamic µs | dyn/static | kernel strategy (static→dynamic) |
|---|---|---:|---:|---:|---|
| var_mean (GroupNorm C=64, spatial s0×s1, hint 16×16) | 8×8   | 6.98  | 9.92   | **1.42×** | persistent→looped, 1→2 kernels |
|  | 16×16 (hint) | 8.42  | 18.05  | **2.14×** | persistent→looped, 1→2 kernels |
|  | 32×32 | 11.87 | 46.82  | **3.94×** | looped→looped, 1→2 kernels |
|  | 64×64 | 32.45 | 167.81 | **5.17×** | looped→looped, 1→2 kernels |
| amax_sum (softmax decomp, (32,128,s2), hint s2=256) | s2=64   | 7.01  | 7.84  | **1.12×** | persistent→looped |
|  | s2=256 (hint) | 7.78  | 9.02  | **1.16×** | persistent→looped |
|  | s2=1024 | 11.10 | 19.39 | **1.75×** | persistent→looped |
|  | s2=4096 | 36.48 | 84.96 | **2.33×** | looped→looped |
| pointwise (gelu chain, (s0,1024), hint s0=64) | s0=16   | 5.76 | 5.50 | **0.96×** | pointwise, 1→1 |
|  | s0=64 (hint) | 5.89 | 5.70 | **0.97×** | pointwise, 1→1 |
|  | s0=256  | 6.98 | 7.04 | **1.01×** | pointwise, 1→1 |
|  | s0=1024 | 7.78 | 9.76 | **1.26×** | pointwise, 1→1 |

## Findings

1. **The cost of dynamism concentrates in reduction-kernel heuristics.**
   Pointwise generalizes essentially for free (≤1.01× until the largest
   point). Reductions pay in two compounding ways, both visible in the
   emitted kernel names:
   - **Persistent-reduction is unavailable dynamically.** Static compiles
     pick `triton_per_*` (persistent) at small numel and switch to
     `triton_red_*` (looped) past the threshold; the general dynamic kernel
     is ALWAYS `red` — the persistent heuristic needs a static reduction
     numel to size its block. At the hint point this alone is ~1.2–2×.
   - **Fusion loss (var_mean only):** the static graph fuses the whole
     GroupNorm body into ONE kernel at every point; the dynamic artifact
     emits TWO (extra materialization + launch). This is the bigger, growing
     term — 5.2× at 64×64.

2. **Dynamic-at-the-hint is already not free for reductions** (2.14× for
   var_mean at its own capture hint). "Compile dynamic, tuned where you
   captured" does not recover static performance; the penalty is structural
   (kernel strategy), not tuning.

3. **The curve degrades with size for reductions** (1.4→5.2×, 1.1→2.3×)
   because the static compiler changes strategy per point while the general
   kernel is pinned to one strategy for the whole family. A floor model that
   prices a dynamic family at its hint-point ratio will UNDERSTATE the cost
   at larger bindings.

## Follow-ups this motivates

- Persistent-reduction dispatch for dynamic kernels (runtime numel branch or
  multi-kernel dispatch) would close most of the small/mid-shape gap —
  connects to the existing persistent-reduction-heuristic work.
- The var_mean 1→2 kernel split under dynamism is a scheduler/fusion
  decision worth its own investigation (why does the symbolic graph break
  the fusion the static graph keeps?).
- Wiring: persist the sweep list + a per-binding reference column
  ({general_us, specialized_us}) in perf.json rows so curves are recoverable
  without re-running (the raw material already exists as ::binding::dynamic
  and ::binding::static rows).
