# sum_sum_sum_f90d684d32cb

## Queue Position

- Rank: 2
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Averroes`
- Closure status: `needs_oracle_measurement`
- Oracle status: `scaffold_measured_not_floor`

## Current Gap

- Best compile: `4347.008228302002 us`
- Memcopy SOL: `119.64800208806992 us`
- Launch-adjusted SOL gap: `31.580612594149382x`
- Oracle path: `repros/canonical/sum_sum_sum_f90d684d32cb/oracle_structured_upsample_reduce.py`
- Measured oracle: `93505.686 us`
- Current best/oracle gap: 0.05x.

## Oracle State

- Measured oracle row exists: oracle_us=93505.686, best_compile_us=4347.008228302002, correct=not_checked, notes=not a true floor; torch-direct scaffold timing
- Next oracle action: measure or replace scaffold with a true optimized canonical oracle before treating it as a floor.

## Inductor Closure Path

- Implementation track: Structured scatter/upsample gather-reduce.
- Candidate hook: Pattern-match scatter/index_put/where feeding reductions and replace dense materialization with output-centric gather-reduce templates.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
