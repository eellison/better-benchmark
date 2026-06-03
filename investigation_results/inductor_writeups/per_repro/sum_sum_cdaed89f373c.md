# sum_sum_cdaed89f373c

## Queue Position

- Rank: 10
- Family: `multi_output_reduction_templates`
- Owner: `Averroes`
- Closure status: `needs_inductor_gap_closure`
- Oracle status: `partial_true_oracle`

## Current Gap

- Best compile: `2486.207962036133 us`
- Memcopy SOL: `611.3280057907104 us`
- Launch-adjusted SOL gap: `3.931832750199308x`
- Oracle path: `repros/canonical/sum_sum_cdaed89f373c/oracle_multi_output_reduction.py`
- Measured oracle: `1492.558 us`
- Current best/oracle gap: 1.67x.

## Oracle State

- Measured oracle row exists: oracle_us=1492.558, best_compile_us=2517.035, correct=not_checked, notes=true fused reduction+epilogue oracle; full upstream version measured 15639.680us, oracle excludes upstream work
- Next oracle action: use measured oracle as the target floor, while preserving notes for partial-graph floors.

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion.
- Candidate hook: Share input reads across same-source reductions, enable scalar accumulators, expand R0_BLOCK candidates, and gate by reduction shape when register pressure regresses siblings.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
