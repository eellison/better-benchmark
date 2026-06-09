# sum_sum_sum_70d71fcb0d68

## Queue Position

- Rank: 26
- Family: `multi_output_reduction_templates`
- Owner: `Noether`
- Closure status: `needs_oracle_measurement`
- Oracle status: `scaffold_measured_not_floor`

## Current Gap

- Best compile: `939.9679899215698 us`
- Memcopy SOL: `154.52800691127777 us`
- Launch-adjusted SOL gap: `5.355088378555368x`
- Oracle path: `repros/canonical/sum_sum_sum_70d71fcb0d68/oracle_cross_dim_reduction.py`
- Measured oracle: `6579.374 us`
- Current best/oracle gap: 0.14x.

## Oracle State

- Measured oracle row exists: oracle_us=6579.374, best_compile_us=939.9679899215698, correct=not_checked, notes=not a true floor; torch-direct scaffold timing
- Next oracle action: measure or replace scaffold with a true optimized canonical oracle before treating it as a floor.

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion.
- Candidate hook: Share input reads across same-source reductions, enable scalar accumulators, expand R0_BLOCK candidates, and gate by reduction shape when register pressure regresses siblings.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
