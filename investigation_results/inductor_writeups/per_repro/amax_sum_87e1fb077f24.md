# amax_sum_87e1fb077f24


## Measured Timings
- Oracle: 371.58 us
- Compile (CDT): 937.66 us
- Ratio: 2.52x

## Queue Position

- Rank: 31
- Family: `online_softmax_cross_entropy`
- Owner: `Fermi`
- Closure status: `scope_mismatch_needs_full_scope_oracle`
- Oracle status: `scope_mismatch_prototype`

## Current Gap

- Best compile: `898.9120125770569 us`
- Memcopy SOL: `162.78399527072906 us`
- Launch-adjusted SOL gap: `4.81257514207335x`
- Oracle path: _none_

## Oracle State

- Existing prototype: `repros/canonical/amax_sum_87e1fb077f24/oracle_online_softmax.py` is diagnosis-only.
- Gap diagnosis: The prototype times a softmax/dropout subset, while the compiled repro includes the broader Longformer attention assembly. Inductor cannot use this as a floor because the benchmark scopes do not match. Classification: `NEW_PATTERN`; the actionable change is unknown until a full-scope Longformer assembly oracle isolates the missing fusion.
- Next oracle action: rewrite only if `--bench` covers the full compiled repro scope; otherwise keep this row out of floor measurements.

## Inductor Closure Path

- Implementation track: Online softmax / CE / softmax-backward templates.
- Candidate hook: Port scalar accumulators + R0_BLOCK expansion, benchmark persistent and looped variants, and gate online templates by row size/reduction pattern.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Full-scope canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
