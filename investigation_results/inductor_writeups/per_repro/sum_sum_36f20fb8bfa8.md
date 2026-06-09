# sum_sum_36f20fb8bfa8


## Measured Timings
- Oracle: 274.18 us
- Compile (CDT): 38.21 us
- Ratio: 0.14x

## Queue Position

- Rank: 713
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `codex`
- Closure status: `implemented_unmeasured`
- Oracle status: `oracle_implemented_needs_measurement`

## Current Gap

- Best compile: `40.064 us`
- Memcopy SOL: `21.280 us`
- Launch-adjusted SOL gap: `1.104x`
- Oracle path: `repros/canonical/sum_sum_36f20fb8bfa8/oracle_index_put_rmsnorm_reduce.py`
- Measured oracle: _pending GPU measurement_
- Shape label: `vllm_qwen_qwen3-30b-a3b_001_3b4ca287`

## Gap Diagnosis

Classification: **SCATTER_REDUCE**

This oracle treats the Qwen MoE grouped-mm row update as an indexed row scatter-add into the `[2048, 2048]` hidden-state matrix and derives the RMSNorm weight-gradient reduction plus transposed input-gradient side output from that single accumulated matrix. Inductor currently materializes the `where` source, lowers `index_put(accumulate=True)` into a generic scatter buffer, then schedules the RMSNorm reduction and transpose-producing pointwise consumer as separate work.

**Root cause**: Inductor's scheduler/codegen does not model duplicate-index row scatter updates as a structured scatter-reduce producer with sibling reduction and materialized-layout epilogues.

**Fix**: Add an indexed row-scatter/RMSNorm-backward lowering that accumulates duplicate expert rows once and fuses the downstream per-hidden reduction with the transposed gradient store.

## Oracle Implementation

- **PyTorch implementation**: `structured_scatter_rmsnorm_reduce` -- performs the full scatter + add + RMSNorm backward chain using standard aten ops, computing weight_grad `[2048]` and transposed_side `[2048, 2048]`.
- **Key insight**: The oracle preserves the exact op sequence but demonstrates that a structured lowering of index_put + RMSNorm backward can avoid redundant reads. The compiled graph's 5 kernels can be collapsed.
- **Total bytes**: 100,827,138 (5 kernels in compiled version)
- **Shape**: BATCH=4, SEQ_LEN=512, HIDDEN=2048

## Inductor Closure Path

- Implementation track: Row-index scatter-reduce template for MoE routing + RMSNorm backward.
- Candidate hook: Detect `where -> index_put(accumulate=True) -> add -> view -> RMSNorm_backward` pattern in post-grad graph; bucket or gather routing rows per token, compute row-local RMSNorm-backward epilogue once, accumulate `[HIDDEN]` weight gradient, and store `[HIDDEN, HIDDEN]` transpose directly.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, and oracle floor.
- Gating policy: gate on MoE routing scatter + RMSNorm backward pattern predicate.

## Done Criteria

- Oracle correctness verified on GPU (--check passes with rtol=0, atol=0).
- Oracle timing measured and appended to measured_oracle_floors.csv.
- Inductor path either reaches the oracle floor or has a gated implementation plan.
