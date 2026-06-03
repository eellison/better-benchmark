# sum_sum_df7b99fa3465

## Queue Position

- Rank: 206
- Family: `multi_output_reduction_templates`
- Owner: `Codex-visformer-bn`
- Closure status: `inductor_gap_writeup_ready`
- Oracle status: `true_oracle_measured`

## Current Gap

- Best compile from queue: `268.0639922618866 us`
- Memcopy SOL: `125.85599720478058 us`
- Launch-adjusted SOL gap: `1.9445218031659968x`
- Oracle path: `repros/canonical/sum_sum_df7b99fa3465/oracle_multi_output_reduction.py`
- Measured oracle: `392.640 us`

## Oracle State

- Full-scope oracle passes `--check`.
- Benchmark command: `python repros/canonical/sum_sum_df7b99fa3465/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `392.640 us`, `coordinate_descent_tuning=True` compile `406.368 us`, combo-looped-CD compile `403.040 us`.

## Gap Diagnosis

The oracle consumes the original repro inputs, fuses the ReLU-mask producer and centered input expression into one split-K pass for the two sibling channel reductions, then reuses those finalized sums in the returned full `[128, 32, 112, 112]` BN-backward epilogue and returned `[32]` scale-gradient vector. Inductor currently schedules the sibling reductions and the dependent full pointwise epilogue as separate graph regions, so it cannot share the masked producer and reduction state across the final materialization. Classification: `SCHEDULER_FUSION`.

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion with dependent full-tensor epilogue.
- Candidate hook: Let the scheduler form a full-scope multi-output reduction template for compatible sibling reductions, then sink the finalized reduction values into the dependent pointwise consumer.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, forced persistent combo, and forced looped combo; gate on exact BN-backward shape if sibling regressions appear.

## Done Criteria

- Inductor path reaches the full-scope oracle or has a measured, gated implementation plan with regression guardrails.
