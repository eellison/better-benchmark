# amax_sum_sum_dc96a87ba8db

## Queue Position

- Rank: 7
- Family: `online_softmax_cross_entropy`
- Owner: `Averroes`
- Closure status: `needs_inductor_gap_closure`
- Oracle status: `true_oracle_measured`

## Current Gap

- Best compile: `3257.344007492065 us`
- Memcopy SOL: `623.6479878425598 us`
- Launch-adjusted SOL gap: `5.173277879681794x`
- Oracle path: `repros/canonical/amax_sum_sum_dc96a87ba8db/oracle_softmax_sum.py`
- Measured oracle: `1638.368010520935 us`
- Current best/oracle gap: 1.99x.

## Oracle State

- Measured oracle row exists: oracle_us=1638.368010520935, best_compile_us=3257.3440074920654, correct=not_checked, notes=Oracle avoids materializing full bf16 softmax; exact repro-equivalent scalar sum within bf16 tolerance.
- Next oracle action: use measured oracle as the target floor, while preserving notes for partial-graph floors.

## Inductor Closure Path

- Implementation track: Online softmax / CE / softmax-backward templates.
- Candidate hook: Port scalar accumulators + R0_BLOCK expansion, benchmark persistent and looped variants, and gate online templates by row size/reduction pattern.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
