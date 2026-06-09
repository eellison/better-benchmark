# sum_b691b8dad90a

## Queue Position

- Rank: 23
- Family: `multi_output_reduction_templates`
- Owner: `unassigned`
- Closure status: `not_true_floor_needs_rewrite`
- Oracle status: `not_true_floor`

## Current Gap

- Best compile: `2091.8400287628174 us` (OLD, pre-fix)
- Memcopy SOL: `687.0719790458679 us`
- Launch-adjusted SOL gap: `3.018214690547139x` (OLD)
- Oracle path: `repros/canonical/sum_b691b8dad90a/oracle_multi_output_reduction.py`

## Re-measurement (2026-06-08)

- Oracle: 877.44 us
- Compiled: 1711.1 us
- Ratio: 1.95x (down from 3.018x)

The decomposed split-K fix (d75864dea06) and aggressive split threshold (8586e404cc8) improved the
compiled code significantly, reducing the gap from 3.02x to 1.95x. The oracle is now a valid floor
(it beats compiled). The remaining 1.95x gap indicates further multi-output reduction fusion work
is needed.

## Oracle State

- Existing prototype: `repros/canonical/sum_b691b8dad90a/oracle_multi_output_reduction.py` is now a valid floor.
- Gap diagnosis: Oracle beats compile (877us vs 1711us). The remaining gap is from multi-output reduction fusion -- sharing input reads across same-source reductions.
- Next oracle action: none needed, oracle establishes valid floor.

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion.
- Candidate hook: Share input reads across same-source reductions, enable scalar accumulators, expand R0_BLOCK candidates, and gate by reduction shape when register pressure regresses siblings.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Full-scope canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
