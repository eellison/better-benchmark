# amax_sum_68fe981b18dd


## Measured Timings
- Oracle: no oracle timing available (scope mismatch)
- Compile (CDT): 776.00 us
- Ratio: N/A

## Queue Position

- Rank: 37
- Family: `online_softmax_cross_entropy`
- Owner: `Fermi`
- Closure status: `scope_mismatch_needs_full_scope_oracle`
- Oracle status: `scope_mismatch_prototype`

## Current Gap

- Best compile: `665.5359864234924 us`
- Memcopy SOL: `93.1520015001297 us`
- Launch-adjusted SOL gap: `4.817418344987695x`
- Oracle path: _none_

## Oracle State

- Existing prototype: `repros/canonical/amax_sum_68fe981b18dd/oracle_online_softmax.py` is diagnosis-only.
- Gap diagnosis: The prototype times the softmax/mask subset, while the compiled repro includes the Longformer attention assembly surrounding that subset. Inductor cannot be credited with a softmax-template miss from this measurement because the oracle excludes the full work region being timed in `repro.py`. Classification: `NEW_PATTERN`; write a full-scope Longformer assembly oracle before assigning a concrete Inductor fix.
- Next oracle action: rewrite only if `--bench` covers the full compiled repro scope; otherwise keep this row out of floor measurements.

## Inductor Closure Path

- Implementation track: Online softmax / CE / softmax-backward templates.
- Candidate hook: Port scalar accumulators + R0_BLOCK expansion, benchmark persistent and looped variants, and gate online templates by row size/reduction pattern.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Full-scope canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
