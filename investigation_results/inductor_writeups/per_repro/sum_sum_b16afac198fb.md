# sum_sum_b16afac198fb

## Queue Position

- Rank: 202
- Family: `multi_output_reduction_templates`
- Owner: `Codex-bn-slice`
- Closure status: `inductor_gap_writeup_ready`
- Oracle status: `true_oracle_measured`

## Current Gap

- Best compile from queue: `222.9440063238144 us`
- Memcopy SOL: `96.44799679517746 us`
- Launch-adjusted SOL gap: `2.241814953628364x`
- Oracle path: `repros/canonical/sum_sum_b16afac198fb/oracle_multi_output_reduction.py`
- Measured oracle: `204.672 us`

## Oracle State

- Full-scope oracle passes `--check`.
- Benchmark command: `python repros/canonical/sum_sum_b16afac198fb/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `204.672 us`, `coordinate_descent_tuning=True` compile `405.536 us`, combo-looped-CD compile `403.424 us`.

## Gap Diagnosis

The oracle consumes the original repro inputs, fuses the ReLU mask, centered producer, sibling channel reductions, and slice-limited BN-backward epilogue, and materializes only the returned `[64, 32, 56, 56]` channel slice plus the `[256]` reduction vector. Inductor currently schedules the sibling reductions, dependent pointwise epilogue, and final channel slice as ordinary graph regions, so it cannot share the masked producer while also sinking the epilogue materialization to the live slice. Classification: `SCHEDULER_FUSION`.

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion with dependent slice-limited epilogue.
- Candidate hook: Let the scheduler form a full-scope multi-output reduction template for compatible sibling reductions, then sink the epilogue materialization to the returned channel slice.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; gate on exact BN-backward shape if sibling regressions appear.

## Done Criteria

- Inductor path reaches the full-scope oracle or has a measured, gated implementation plan with regression guardrails.
