# amax_sum_amax_68fa105ccaf0

## Queue Position

- Rank: 38
- Family: `online_softmax_cross_entropy`
- Owner: `Fermi`
- Closure status: `needs_oracle_measurement`
- Oracle status: `queued`

## Current Gap

- Best compile: `804.8639893531799 us`
- Memcopy SOL: `160.67199409008026 us`
- Launch-adjusted SOL gap: `4.829029578407634x`
- Oracle path: `repros/canonical/amax_sum_amax_68fa105ccaf0/oracle_online_softmax.py`

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
