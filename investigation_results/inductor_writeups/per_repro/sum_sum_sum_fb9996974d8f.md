# sum_sum_sum_fb9996974d8f

## Queue Position

- Rank: 199
- Family: `multi_output_reduction_templates`
- Owner: `unassigned`
- Closure status: `not_true_floor_needs_rewrite`
- Oracle status: `not_true_floor`

## Current Gap

- Best compile from queue: `270.3039944171905 us`
- Memcopy SOL: `117.18399822711945 us`
- Launch-adjusted SOL gap: `1.9145512084334x`
- Oracle path: _none_

## Oracle State

- Existing full-scope diagnosis artifact: `repros/canonical/sum_sum_sum_fb9996974d8f/oracle_multi_output_reduction.py`.
- The script passes `--check` and preserves output strides, but it is not a floor.
- Benchmark command: `python repros/canonical/sum_sum_sum_fb9996974d8f/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `630.816 us`, `coordinate_descent_tuning=True` compile `454.784 us`, combo-looped-CD compile `444.256 us`.

## Gap Diagnosis

The diagnosis oracle covers the full add, copy/clone/slice, and two BN-backward dual-reduction regions, but its split schedule is slower than tuned Inductor. Inductor still lacks a clean memory-format-aware multi-output reduction template for this pattern, but the current oracle implementation does not establish an achievable floor and must not be wired as one. Classification: `SCHEDULER_FUSION`.

## Inductor Closure Path

- Implementation track: Memory-format-aware multi-output reduction fusion.
- Candidate hook: Share the add producer across the C=24 and C=12 sibling reductions while preserving the returned output layouts.
- Next oracle action: rewrite a faster full-scope oracle or close the row if tuned compile is the realistic floor.

## Done Criteria

- Faster full-scope oracle measured, or blocker documented with tuned compile treated as the realistic floor.
