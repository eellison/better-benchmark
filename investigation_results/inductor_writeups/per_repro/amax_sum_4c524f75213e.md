# amax_sum_4c524f75213e


## Measured Timings
- Oracle: no oracle timing available (scope mismatch)
- Compile (CDT): 962.59 us
- Ratio: N/A

## Queue Position

- Rank: 21
- Family: `online_softmax_cross_entropy`
- Owner: `Averroes`
- Closure status: `scope_mismatch_needs_full_scope_oracle`
- Oracle status: `scope_mismatch_prototype`

## Current Gap

- Best compile: `867.6159977912903 us`
- Memcopy SOL: `93.34400296211244 us`
- Launch-adjusted SOL gap: `6.138329038437214x`
- Oracle path: _none_

## Oracle State

- Existing prototype: `repros/canonical/amax_sum_4c524f75213e/oracle_online_softmax.py` is diagnosis-only.
- Gap diagnosis: The prototype times only the online softmax core, while the compiled repro spends most of its time in Longformer attention assembly around that core. This cannot be claimed as an Inductor softmax-template gap because the current oracle excludes the dominant surrounding work. Classification: `NEW_PATTERN`; write a full-scope Longformer attention assembly oracle before assigning an Inductor fix.
- Next oracle action: rewrite only if `--bench` covers the full compiled repro scope; otherwise keep this row out of floor measurements.

## Inductor Closure Path

- Implementation track: Online softmax / CE / softmax-backward templates.
- Candidate hook: Port scalar accumulators + R0_BLOCK expansion, benchmark persistent and looped variants, and gate online templates by row size/reduction pattern.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Full-scope canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
