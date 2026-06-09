# amax_sum_f0661488d68c


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 1007.52 us
- Ratio: N/A

## Queue Position

- Rank: 14
- Family: `online_softmax_cross_entropy`
- Owner: `Averroes`
- Closure status: `needs_oracle_measurement`
- Oracle status: `oracle_scaffolded`

## Current Gap

- Best compile: `2225.055932998657 us`
- Memcopy SOL: `625.5040168762207 us`
- Launch-adjusted SOL gap: `3.540241387887368x`
- Oracle path: `repros/canonical/amax_sum_f0661488d68c/oracle_cross_entropy_fwd.py`

## Oracle State

- No measured oracle row yet.
- Next oracle action: measure or replace scaffold with a true optimized canonical oracle before treating it as a floor.

## Inductor Closure Path

- Implementation track: Online softmax / CE / softmax-backward templates.
- Candidate hook: Port scalar accumulators + R0_BLOCK expansion, benchmark persistent and looped variants, and gate online templates by row size/reduction pattern.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
