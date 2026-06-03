# sum_sum_100fff569d2e

## Queue Position

- Rank: 208
- Family: `multi_output_reduction_templates`
- Owner: `Codex-densenet-slice2`
- Closure status: `inductor_gap_writeup_ready`
- Oracle status: `true_oracle_measured`

## Current Gap

- Best compile from queue: `210.91200411319733 us`
- Memcopy SOL: `91.13600105047226 us`
- Launch-adjusted SOL gap: `2.171306228713354x`
- Oracle path: `repros/canonical/sum_sum_100fff569d2e/oracle_multi_output_reduction.py`
- Measured oracle: `191.648 us`

## Oracle State

- Full-scope oracle passes `--check`.
- Benchmark command: `python repros/canonical/sum_sum_100fff569d2e/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `191.648 us`, `coordinate_descent_tuning=True` compile `382.880 us`, combo-looped-CD compile `378.016 us`.

## Gap Diagnosis

The oracle consumes the original repro inputs, fuses the ReLU-mask producer, centered input expression, and two sibling channel reductions across all 224 BN channels, then materializes only the returned channels 192:224 of the dependent BN-backward epilogue while adding the residual slice from `mul_972`. Inductor currently schedules the residual slice, sibling reductions, dependent pointwise epilogue, and final slice/add as separate graph regions, so it cannot share the masked producer and reduce state while limiting materialization to the live channel slice. Classification: `SCHEDULER_FUSION`.

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion with dependent residual slice epilogue.
- Candidate hook: Let the scheduler form a full-scope multi-output reduction template for compatible sibling reductions, then sink the dependent epilogue and residual add to the returned channel slice.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; gate on exact BN-backward slice shapes if sibling regressions appear.

## Done Criteria

- Inductor path reaches the full-scope oracle or has a measured, gated implementation plan with regression guardrails.
