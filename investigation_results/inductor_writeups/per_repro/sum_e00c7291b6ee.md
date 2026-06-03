# sum_e00c7291b6ee

## Queue Position

- Rank: 24
- Family: `multi_output_reduction_templates`
- Owner: `unassigned`
- Closure status: `not_true_floor_needs_rewrite`
- Oracle status: `not_true_floor`

## Current Gap

- Best compile: `2984.9600791931152 us`
- Memcopy SOL: `1248.35205078125 us`
- Launch-adjusted SOL gap: `2.385387930862087x`
- Oracle path: _none_

## Oracle State

- Existing prototype: `repros/canonical/sum_e00c7291b6ee/oracle_multi_output_reduction.py` is not a valid floor.
- Gap diagnosis: The script attempts the reduction oracle scope but is slower than the compiled baseline, so it does not prove a reachable floor. Inductor cannot be optimized against this measurement because the oracle has worse parallelism/tiling than generated code. Classification: `COOPERATIVE_SPLIT_K`; a valid oracle/fix needs split-K style parallel reduction with safe atomic coordination or another full-scope strategy that actually beats compile.
- Next oracle action: rewrite before treating this as a floor.

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion.
- Candidate hook: Share input reads across same-source reductions, enable scalar accumulators, expand R0_BLOCK candidates, and gate by reduction shape when register pressure regresses siblings.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Full-scope canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
