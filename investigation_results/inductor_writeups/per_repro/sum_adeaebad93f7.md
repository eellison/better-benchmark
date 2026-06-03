# sum_adeaebad93f7

## Queue Position

- Rank: 5
- Family: `irregular_gather_reduce`
- Owner: `Confucius`
- Closure status: `oracle_active_before_gap_closure`
- Oracle status: `active_oracle_measurement`

## Current Gap

- Best compile: `2434.0479373931885 us`
- Memcopy SOL: `160.60799360275269 us`
- Launch-adjusted SOL gap: `14.351021350409304x`
- Oracle path: `repros/canonical/sum_adeaebad93f7/oracle_fused_gather_reduce.py`

## Oracle State

- No measured oracle row yet.
- Next oracle action: use measured oracle as the target floor, while preserving notes for partial-graph floors.

## Inductor Closure Path

- Implementation track: Irregular gather-reduce specialization.
- Candidate hook: Special-case one-hot/gather/cat/reduce graphs only when index structure is provably reusable.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
