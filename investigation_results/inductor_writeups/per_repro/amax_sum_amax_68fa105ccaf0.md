# amax_sum_amax_68fa105ccaf0

## Queue Position

- Rank: 38
- Family: `online_softmax_cross_entropy`
- Owner: `Fermi`
- Closure status: `scope_mismatch_needs_full_scope_oracle`
- Oracle status: `scope_mismatch_prototype`

## Current Gap

- Best compile: `804.8639893531799 us`
- Memcopy SOL: `160.67199409008026 us`
- Launch-adjusted SOL gap: `4.829029578407634x`
- Oracle path: _none_

## Oracle State

- Existing prototype: `repros/canonical/amax_sum_amax_68fa105ccaf0/oracle_online_softmax.py` is diagnosis-only.
- Gap diagnosis: The prototype times a dual softmax subset and misses the T5 position-bias, mask, and dropout work in the compiled repro. Inductor cannot use this as a scalar-accumulator floor because the oracle excludes operations that must stay in scope. Classification: `SCHEDULER_FUSION`; a valid Inductor change would expand the scheduled/template region around online softmax so the surrounding pointwise/dropout work is fused or accounted for.
- Next oracle action: rewrite full-scope oracle before treating this as a floor.

## Inductor Closure Path

- Implementation track: Online softmax / CE / softmax-backward templates.
- Candidate hook: Port scalar accumulators + R0_BLOCK expansion, benchmark persistent and looped variants, and gate online templates by row size/reduction pattern.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Full-scope canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
