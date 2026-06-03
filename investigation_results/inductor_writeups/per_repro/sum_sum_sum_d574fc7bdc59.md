# sum_sum_sum_d574fc7bdc59

## Queue Position

- Rank: 545
- Family: `multi_output_reduction_templates`
- Owner: `unassigned`
- Closure status: `not_true_floor_needs_rewrite`
- Oracle status: `full_scope_measured_not_floor`

## Current Gap

- Best compile from queue: `46.78399860858917 us`
- Memcopy SOL: `21.56800031661988 us`
- Launch-adjusted SOL gap: `1.3937082390167257x`
- Diagnosis artifact: `repros/canonical/sum_sum_sum_d574fc7bdc59/oracle_multi_output_reduction.py`
- Measured oracle: `130.368 us`

## Oracle State

- Full-scope oracle passes `--check`, including output stride checks.
- Benchmark command: `python repros/canonical/sum_sum_sum_d574fc7bdc59/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `130.368 us`, `coordinate_descent_tuning=True` compile `74.528 us`, combo-looped-CD compile `77.824 us`.
- This artifact is intentionally not wired as `oracle_path` in the main queue because it is slower than tuned compile.

## Gap Diagnosis

The oracle consumes the original Whisper inputs, computes row-local normalization backward summaries, accumulates the two sibling `[384]` column reductions, and writes the returned full `[8, 1500, 384]` residual-add tensor. Inductor's tuned kernels are already substantially faster than this three-kernel hand oracle, but the remaining actionable shape is still the same: coordinating row-parallel work with two column reductions and a live full-tensor side output. Classification: `COOPERATIVE_SPLIT_K`.

## Inductor Closure Path

- Implementation track: Cooperative split-K reduction with materialized side output, only if it beats the existing tuned schedule.
- Candidate hook: Coordinate partial reductions while producing the full epilogue output, without adding extra passes over `mm_6 + mm_9 + mm_10`.
- Benchmark policy: keep this row in rewrite-needed state until a faster full-scope oracle or Inductor implementation beats `coordinate_descent_tuning=True`.
