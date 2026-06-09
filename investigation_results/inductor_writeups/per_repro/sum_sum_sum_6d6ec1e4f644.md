# sum_sum_sum_6d6ec1e4f644

## Queue Position

- Family: `multi_output_reduction_templates`
- Owner: `Codex-bottom-gpt2-ln`
- Oracle path: `repros/canonical/sum_sum_sum_6d6ec1e4f644/oracle_multi_output_reduction.py`

## Oracle State

- Full-scope oracle passes `--check`, including stride checks for all three `[768]` outputs.
- Benchmark command: `python repros/canonical/sum_sum_sum_6d6ec1e4f644/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `39.712 us`, `coordinate_descent_tuning=True` compile `51.552 us`, combo-looped-CD compile `54.752 us`.

## Gap Diagnosis

The oracle covers the full GPT-2 layernorm/dropout backward repro: it consumes the same inputs as `repro.py`, applies the `mm_1` reshape, multiplies by `arg73_1`, computes the per-row C=768 layernorm reductions, applies the dependent pointwise expression with `arg233_1` and the scaled dropout mask, and accumulates all three returned `[768]` column reductions. Inductor currently materializes or separately schedules the row reductions and final column reduction because the final output depends on reductions over a different axis than the returned sibling column sums. Classification: `COOPERATIVE_SPLIT_K`.

## Inductor Closure Path

- Implementation track: cooperative split-K multi-output reduction for dependent row-local reductions feeding column accumulators.
- Candidate hook: let the scheduler/codegen form row tiles that compute the row-local scalar reductions and accumulate all compatible column outputs into partial buffers or atomics.
- Benchmark policy: compare against default, `coordinate_descent_tuning=True`, and the forced combo-looped-CD config on this exact C=768 GPT-2 shape.
