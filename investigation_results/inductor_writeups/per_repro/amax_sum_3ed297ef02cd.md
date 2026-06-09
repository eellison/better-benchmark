# amax_sum_3ed297ef02cd

## Queue Position

- Rank: 12
- Family: `online_softmax_cross_entropy`
- Owner: `Averroes`
- Closure status: `needs_inductor_gap_closure`
- Oracle status: `true_oracle_measured`

## Current Gap

- Best compile: `3494.080066680908 us`
- Memcopy SOL: `1243.1679964065552 us`
- Launch-adjusted SOL gap: `2.803859573313087x`
- Oracle path: `repros/canonical/amax_sum_3ed297ef02cd/oracle_online_softmax.py`
- Measured oracle: `1942.9 us`
- Current best/oracle gap: 1.80x.

## Oracle State

- Measured oracle row exists: oracle_us=1942.9, best_compile_us=2078.9, correct=true, notes=true Triton online softmax oracle; script auto-append normalized after schema mismatch
- Next oracle action: use measured oracle as the target floor, while preserving notes for partial-graph floors.

## Inductor Closure Path

- Implementation track: Online softmax / CE / softmax-backward templates.
- Candidate hook: Port scalar accumulators + R0_BLOCK expansion, benchmark persistent and looped variants, and gate online templates by row size/reduction pattern.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; prioritize best runtime over default heuristic purity.
- Gating policy: if the optimization closes this repro but regresses a sibling, gate on the exact pattern/shape/reduction-size predicate and keep both paths autotuned.

## Done Criteria

- Canonical oracle measured or blocker documented.
- Inductor path either reaches the oracle/realistic floor or has a measured, gated implementation plan with regression guardrails.
