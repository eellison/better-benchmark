# pointwise_70c0751eb408

## Queue Position

- Rank: 17
- Family: `layout_indexing_stencil_fusion`
- Owner: `Maxwell`
- Closure status: `needs_oracle_measurement`
- Oracle status: `implemented`

## Current Gap

- Best compile: `939.7760033607484 us`
- Memcopy SOL: `74.49600100517273 us`
- Launch-adjusted SOL gap: `3.411214681635625x`
- Oracle path: `repros/canonical/pointwise_70c0751eb408/oracle_stencil_canonicalized.py`

## Oracle State

- No measured oracle row yet.
- Next oracle action: measure or replace scaffold with a true optimized canonical oracle before treating it as a floor.

## Inductor Closure Path

- Implementation track: Layout/stencil materialization elision.
- Candidate hook: Extend slice_scatter/select_scatter/stencil canonicalization and gate by exact alias-safe index region checks.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
