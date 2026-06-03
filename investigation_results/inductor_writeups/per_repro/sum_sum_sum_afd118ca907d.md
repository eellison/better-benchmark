# sum_sum_sum_afd118ca907d

## Queue Position

- Rank: 3
- Family: `online_softmax_cross_entropy`
- Owner: `Confucius`
- Closure status: `oracle_active_before_gap_closure`
- Oracle status: `active_oracle_measurement`

## Current Gap

- Best compile: `6364.192008972168 us`
- Memcopy SOL: `654.2080044746399 us`
- Launch-adjusted SOL gap: `9.46759331428382x`
- Oracle path: `repros/canonical/sum_sum_sum_afd118ca907d/oracle_softmax_backward.py`

## Oracle State

- No measured oracle row yet.
- Next oracle action: use measured oracle as the target floor, while preserving notes for partial-graph floors.

## Inductor Closure Path

- Implementation track: Online softmax / CE / softmax-backward templates.
- Candidate hook: Port scalar accumulators + R0_BLOCK expansion, benchmark persistent and looped variants, and gate online templates by row size/reduction pattern.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
