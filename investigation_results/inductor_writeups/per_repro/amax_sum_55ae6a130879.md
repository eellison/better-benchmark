# amax_sum_55ae6a130879


## Measured Timings
- Oracle: 113.00 us
- Compile (CDT): 165.50 us
- Ratio: 1.46x

## Queue Position

- Rank: 197
- Family: `online_softmax_cross_entropy`
- Owner: `unassigned`
- Closure status: `scope_mismatch_needs_full_scope_oracle`
- Oracle status: `scope_mismatch_prototype`

## Current Gap

- Best compile: `175.87199807167053 us`
- Memcopy SOL: `64.41599875688553 us`
- Launch-adjusted SOL gap: `2.6087575844703754x`
- Oracle path: _none_

## Prototype Notes

- Existing prototype: `repros/canonical/amax_sum_55ae6a130879/oracle_online_softmax.py` is diagnosis-only.
- Prior B200 measurement with coordinate-descent tuning reported compile around `179 us`, prototype around `113 us`, and one generated kernel.
- The prototype note also found `scalar_reduction_accumulators` does not matter here because `rnumel=512` already fits in a persistent reduction tile.

## Gap Diagnosis

The prototype measures a softmax/dropout core, while the compiled repro includes fused dropout plus permute work around that core. The prototype also avoids part of the real dropout/RNG and layout path, so it is not a full-scope floor for the compiled `repro.py` region. Inductor cannot use that number as a softmax-reduction target because the reduction strategy is already persistent; the remaining gap is in keeping the surrounding attention assembly work fused and indexed compactly.

Classification: `SCHEDULER_FUSION`. A valid fix would keep the surrounding dropout/permute work in the same full-scope optimized path, or a replacement oracle must benchmark that full path before the row can be counted as a floor.

## Inductor Closure Path

- Implementation track: Attention softmax/dropout/layout template recognition.
- Candidate hook: Expand the online-softmax template or scheduler fusion boundary so dropout and final layout indexing stay in the optimized full-scope path.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: gate on the exact attention shape/layout/reduction-size predicate and keep both paths autotuned if sibling regressions appear.

## Done Criteria

- Full-scope canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
