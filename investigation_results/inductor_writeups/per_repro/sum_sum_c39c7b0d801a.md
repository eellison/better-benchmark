# sum_sum_c39c7b0d801a

## Queue Position

- Rank: 689
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `codex`
- Closure status: `implemented_unmeasured`
- Oracle status: `oracle_implemented_needs_measurement`

## Current Gap

- Best compile: `42.912 us`
- Memcopy SOL: `23.200 us`
- Launch-adjusted SOL gap: `1.123x`
- Oracle path: `repros/canonical/sum_sum_c39c7b0d801a/oracle_moe_index_put_reduce.py`
- Measured oracle: _pending GPU measurement_
- Shape label: `vllm_qwen_qwen3-30b-a3b_001_f65095c5`

## Gap Diagnosis

Classification: **SCATTER_REDUCE**

This oracle accumulates the Qwen MoE row-index `index_put(accumulate=True)` into logical `[2048, 2048]` token rows and immediately reuses those rows for the RMSNorm weight-gradient sum plus the returned transposed input-gradient store. Inductor currently materializes the BF16 row-scatter result, adds `mm_23`, then schedules the `[0, 1]` weight-gradient reduction and RMSNorm-backward/permute side output as generic consumers.

**Root cause**: Inductor's scheduler/codegen does not recognize a one-dimensional MoE routing-index scatter feeding row-local RMSNorm backward reductions as a structured scatter-reduce with a required layout-changing side store.

**Fix**: Add a post-grad row-index scatter-reduce template that buckets or gathers routing rows per token, computes the row-local RMSNorm-backward epilogue once, accumulates the `[2048]` weight gradient, and stores the `[2048, 2048]` transpose directly.

## Oracle Implementation

- **PyTorch implementation**: `oracle_moe_index_put_reduce` -- performs `row_index_accumulate` (where + index_put with accumulate), adds mm_23, reshapes, then computes RMSNorm backward (weight_grad reduction and input_grad transpose).
- **Key insight**: Sibling of sum_sum_36f20fb8bfa8 with same MoE routing pattern but different shape config. Demonstrates the same scatter-reduce template opportunity.
- **Total bytes**: 109,215,746 (5 kernels in compiled version)
- **Shape**: HIDDEN_SIZE=2048

## Inductor Closure Path

- Implementation track: Row-index scatter-reduce template for MoE routing + RMSNorm backward (shared with sum_sum_36f20fb8bfa8).
- Candidate hook: Same pattern as sum_sum_36f20fb8bfa8: detect `where -> index_put(accumulate=True) -> add -> view -> RMSNorm_backward` in post-grad graph.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, and oracle floor.
- Gating policy: gate on MoE routing scatter + RMSNorm backward pattern predicate.

## Done Criteria

- Oracle correctness verified on GPU (--check passes with rtol=2e-2, atol=2e-1).
- Oracle timing measured and appended to measured_oracle_floors.csv.
- Inductor path either reaches the oracle floor or has a gated implementation plan.
