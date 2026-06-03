# var_mean_765fb8f2c85e

## Queue Position

- Rank: 29
- Family: `norm_template_canonicalization`
- Owner: `Confucius`
- Closure status: `oracle_active_before_gap_closure`
- Oracle status: `active_oracle_measurement`

## Current Gap

- Best compile: `746.4960217475891 us`
- Memcopy SOL: `99.2640033364296 us`
- Launch-adjusted SOL gap: `6.533081285010357x`
- Oracle path: `repros/canonical/var_mean_765fb8f2c85e/oracle_bn_training_forward.py`

## Oracle State

- No measured oracle row yet.
- Next oracle action: use measured oracle as the target floor, while preserving notes for partial-graph floors.

## Inductor Closure Path

- Implementation track: Norm/pool composite templates.
- Candidate hook: Recover BN/LN/RMSNorm composite templates after decomposition; tune warps/blocks and gate when generic reductions already win.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
