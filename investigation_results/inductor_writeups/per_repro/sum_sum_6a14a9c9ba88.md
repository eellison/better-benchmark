# sum_sum_6a14a9c9ba88

## Queue Position

- Rank: 11
- Family: `multi_output_reduction_templates`
- Owner: `Averroes`
- Closure status: `needs_inductor_gap_closure`
- Oracle status: `partial_true_oracle`

## Current Gap

- Best compile: `1626.1759996414185 us`
- Memcopy SOL: `242.5920069217682 us`
- Launch-adjusted SOL gap: `6.099867803308229x`
- Oracle path: `repros/canonical/sum_sum_6a14a9c9ba88/oracle_multi_output_reduction.py`
- Measured oracle: `2491.8 us`
- Current best/oracle gap: 0.65x.

## Oracle State

- Measured oracle row exists: oracle_us=2491.8, best_compile_us=3394.4, correct=not_checked, notes=partial true oracle for fused dual-reduction+post-reduction pointwise; scatter_add/maxpool backward precomputed
- Next oracle action: use measured oracle as the target floor, while preserving notes for partial-graph floors.

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion.
- Candidate hook: Share input reads across same-source reductions, enable scalar accumulators, expand R0_BLOCK candidates, and gate by reduction shape when register pressure regresses siblings.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
