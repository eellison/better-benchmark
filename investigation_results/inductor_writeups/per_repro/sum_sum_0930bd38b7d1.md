# sum_sum_0930bd38b7d1


## Measured Timings
- Oracle: 944.10 us
- Compile (CDT): 1254.40 us
- Ratio: 1.33x

## Queue Position

- Rank: 20
- Family: `multi_output_reduction_templates`
- Owner: `unassigned`
- Closure status: `scope_mismatch_needs_full_scope_oracle`
- Oracle status: `scope_mismatch_prototype`

## Current Gap

- Best compile: `1253.216028213501 us`
- Memcopy SOL: `240.06399512290955 us`
- Launch-adjusted SOL gap: `4.692568264908714x`
- Oracle path: _none_

## Oracle State

- Existing prototype: `repros/canonical/sum_sum_0930bd38b7d1/oracle_multi_output_reduction.py` is diagnosis-only.
- Gap diagnosis: The prototype precomputes scatter/pool/BN setup and times only the reduction-plus-pointwise tail; the compiled repro is already faster for the full region. Inductor cannot use this as a floor because the oracle removes work that Inductor still has to perform. Classification: `SCATTER_REDUCE`; a valid fix must eliminate the scatter intermediate through a full-scope gather-reduce, not by timing a precomputed subset.
- Next oracle action: rewrite full-scope oracle or leave this row actionable without a floor.

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion.
- Candidate hook: Share input reads across same-source reductions, enable scalar accumulators, expand R0_BLOCK candidates, and gate by reduction shape when register pressure regresses siblings.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Full-scope canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
