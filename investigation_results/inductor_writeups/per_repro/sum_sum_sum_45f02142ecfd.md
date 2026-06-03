# sum_sum_sum_45f02142ecfd

## Queue Position

- Rank: 1
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Kepler`
- Closure status: `oracle_active_before_gap_closure`
- Oracle status: `active_oracle_impl`

## Current Gap

- Best compile: `14377.087593078611 us`
- Memcopy SOL: `232.35200345516205 us`
- Launch-adjusted SOL gap: `54.18119104387311x`
- Oracle path: `repros/canonical/sum_sum_sum_45f02142ecfd/oracle_structured_scatter_reduce.py`

## Oracle State

- No measured oracle row yet.
- Next oracle action: use measured oracle as the target floor, while preserving notes for partial-graph floors.

## Inductor Closure Path

- Implementation track: Structured scatter/upsample gather-reduce.
- Candidate hook: Pattern-match scatter/index_put/where feeding reductions and replace dense materialization with output-centric gather-reduce templates.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
