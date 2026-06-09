# sum_sum_8bcd6e12dcd4


## Measured Timings
- Oracle: 5006.73 us
- Compile (CDT): 792.45 us
- Ratio: 0.16x

## Queue Position

- Rank: 18
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Averroes`
- Closure status: `needs_oracle_measurement`
- Oracle status: `queued`

## Current Gap

- Best compile: `782.4320197105408 us`
- Memcopy SOL: `70.17599791288376 us`
- Launch-adjusted SOL gap: `8.87352610949262x`
- Oracle path: `repros/canonical/sum_sum_8bcd6e12dcd4/oracle_structured_scatter_reduce.py`

## Oracle State

- Measured oracle row exists: oracle_us=5006.73393253237, best_compile_us=782.4320197105408, correct=not_checked, notes=Fused scatter-reduce oracle: bypasses [65536,3025] intermediate via gather-mask-reduce.
- Next oracle action: measure or replace scaffold with a true optimized canonical oracle before treating it as a floor.

## Inductor Closure Path

- Implementation track: Structured scatter/upsample gather-reduce.
- Candidate hook: Pattern-match scatter/index_put/where feeding reductions and replace dense materialization with output-centric gather-reduce templates.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
