# var_mean_f0d7c08a0622


## Measured Timings
- Oracle: 946.02 us
- Compile (CDT): 909.15 us
- Ratio: 0.96x

## Queue Position

- Rank: 25
- Family: `norm_template_canonicalization`
- Owner: `Averroes`
- Closure status: `needs_oracle_measurement`
- Oracle status: `queued`

## Current Gap

- Best compile: `909.1519713401794 us`
- Memcopy SOL: `140.03199338912964 us`
- Launch-adjusted SOL gap: `5.980004281159312x`
- Oracle path: `repros/canonical/var_mean_f0d7c08a0622/oracle_norm_template.py`

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
