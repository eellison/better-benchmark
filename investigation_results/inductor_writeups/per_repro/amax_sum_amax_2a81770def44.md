# amax_sum_amax_2a81770def44

## Queue Position

- Rank: 22
- Family: `online_softmax_cross_entropy`
- Owner: `unassigned`
- Closure status: `scope_mismatch_needs_full_scope_oracle`
- Oracle status: `scope_mismatch_prototype`

## Current Gap

- Best compile: `1219.391942024231 us`
- Memcopy SOL: `238.5919988155365 us`
- Launch-adjusted SOL gap: `4.985412229056019x`
- Oracle path: _none_

## Oracle State

- Existing prototype: `repros/canonical/amax_sum_amax_2a81770def44/oracle_online_softmax.py` is diagnosis-only.
- Gap diagnosis: The prototype times the dual online softmax/dropout core but omits the surrounding T5 position-bias, mask, and dropout scope in the compiled repro. Inductor cannot close the measured full gap with only the scalar-accumulator softmax kernel because the scheduler/template boundary excludes surrounding pointwise/dropout work. Classification: `SCHEDULER_FUSION`; a valid fix would expand the scheduled/template region to keep those surrounding ops in the full softmax path.
- Next oracle action: rewrite full-scope oracle before treating this as a floor.

## Inductor Closure Path

- Implementation track: Online softmax / CE / softmax-backward templates.
- Candidate hook: Port scalar accumulators + R0_BLOCK expansion, benchmark persistent and looped variants, and gate online templates by row size/reduction pattern.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Full-scope canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
