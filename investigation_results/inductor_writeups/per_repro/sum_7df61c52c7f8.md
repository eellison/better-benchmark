# sum_7df61c52c7f8

## Queue Position

- Rank: 30
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Noether`
- Closure status: `needs_oracle_measurement`
- Oracle status: `queued`

## Current Gap

- Best compile: `636.9600296020508 us`
- Memcopy SOL: `69.69600170850754 us`
- Launch-adjusted SOL gap: `7.796709952522927x`
- Oracle path: `repros/canonical/sum_7df61c52c7f8/oracle_multi_output_reduction.py`

## Oracle State

- No measured oracle row yet.
- Next oracle action: measure or replace scaffold with a true optimized canonical oracle before treating it as a floor.

## Inductor Closure Path

- Implementation track: Structured scatter/upsample gather-reduce.
- Candidate hook: Pattern-match scatter/index_put/where feeding reductions and replace dense materialization with output-centric gather-reduce templates.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
