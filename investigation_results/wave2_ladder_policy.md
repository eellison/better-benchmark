# Wave-2 ladder policy (settles plan §9 open question 6)

Decision date: 2026-06-12. Scope: which symbol bindings each dynamic
family is benchmarked at. Capture records only observed points
(`source: "captured"`); ladder points (`source: "ladder"`) are added by
BENCH POLICY, here — never by capture, never by hand-editing shapes.json
(a generator script materializes this policy; see "Mechanics").

## Principles

1. **The hint point is always measured** — it is the static-corpus
   anchor: the binding where static-vs-dynamic compile is an
   apples-to-apples comparison (same concrete shapes, both artifacts).
2. **Ladders are per-symbol-ROLE, not per-model.** A symbol's role
   (batch, sequence length, spatial, vocab/feature) determines its
   ladder. Roles are inferred from the source model + input position and
   recorded in the policy output, reviewable per family.
3. **Geometric ladders, capped count.** Kernel behavior shifts at
   power-of-two occupancy boundaries; intermediate points add bench time
   without changing conclusions. Cap: ≤6 ladder points per symbol, ≤12
   total points per pattern (beyond that, runtime dominates and the
   marginal point answers nothing).
4. **Respect ranges and guards.** Every ladder point must pass
   validate_bindings (range + residual guards) — the codec enforces this
   loudly; policy never overrides it.
5. **Both compile modes at every point.** Each (binding) row is measured
   static-compiled (specialize at that binding) AND with the shared
   dynamic artifact — the static-vs-dynamic gap IS the wave-2 question
   (vllm use case: is dynamic=True's looped kernel an acceptable tax).

## Ladders by role

| Role | Ladder | Rationale |
|---|---|---|
| decode batch (vllm BS) | 1, 2, 4, 8, 16, 32, 64 (∩ range) | plan §9 suggestion; covers latency→throughput regimes |
| prefill / LM sequence | 128, 512, 1024, 2048, 4096, 8192 (∩ range) | crosses persistent→looped reduction boundaries |
| spatial (H/W, opacus-class) | hint/2, hint, 2*hint (∩ range) | conv-adjacent shapes rarely stray far from trained size |
| feature/vocab/head dims | hint only | architecturally fixed; dynamism is incidental capture noise |
| unknown role | hint, plus range-clamped {hint/2, 2*hint} | conservative default; flag for review |

Coupled symbols (one expr references several, e.g. s16*s82): bind
together along their joint observed direction (square spatial ladders
bind both), never a cartesian product — cartesian explodes the cap and
off-diagonal points are unobserved hypotheticals.

## Mechanics

- A script (`scripts/generate_ladder_points.py`, to be written when the
  capture side lands) reads each dynamic pattern's symbols/guards,
  assigns roles (model-name + input-position heuristics, overridable via
  a checked-in per-family YAML/JSON override file), and emits ladder
  bindings as bench-time inputs — NOT written into the pattern's
  shapes.json (capture artifacts stay capture-only; policy output lives
  in investigation_results/ or is passed via --bind).
- bench side already supports this: `--bind s0=4,s1=1` repeatable rows +
  `--dynamic` compile-once measurement (landed 2026-06-12).
- Oracles: dispatch stays exact-point. Oracle coverage of ladder points
  is a SEPARATE queue decision (an oracle measured at the hint is not
  presumed the floor at BS=64); ladder rows without an oracle get
  compile-vs-compile (static vs dynamic) numbers only.
