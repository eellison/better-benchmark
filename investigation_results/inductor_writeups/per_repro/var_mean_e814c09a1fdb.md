# var_mean_e814c09a1fdb


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 6982.62 us
- Ratio: N/A

## Queue Position

- Rank: 27
- Family: `norm_template_canonicalization`
- Owner: `Noether`
- Closure status: `needs_oracle_measurement`
- Oracle status: `queued`

## Current Gap

- Best compile: `1778.6879539489746 us`
- Memcopy SOL: `594.111979007721 us`
- Launch-adjusted SOL gap: `2.9201329398357285x`
- Oracle path: `repros/canonical/var_mean_e814c09a1fdb/oracle_norm_template.py`

## Oracle State

- No measured oracle row yet.
- Next oracle action: measure or replace scaffold with a true optimized canonical oracle before treating it as a floor.

## Inductor Closure Path

- Implementation track: Norm/pool composite templates.
- Candidate hook: Recover BN/LN/RMSNorm composite templates after decomposition; tune warps/blocks and gate when generic reductions already win.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
