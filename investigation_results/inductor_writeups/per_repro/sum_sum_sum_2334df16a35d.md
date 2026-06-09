# sum_sum_sum_2334df16a35d


## Measured Timings
- Oracle: 44.32 us
- Compile (CDT): 46.91 us
- Ratio: 1.06x

## Queue Position

- Rank: 778
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `codex`
- Closure status: `implemented_unmeasured`
- Oracle status: `oracle_implemented_needs_measurement`

## Current Gap

- Best compile: `48.864 us`
- Memcopy SOL: `29.408 us`
- Launch-adjusted SOL gap: `1.031x`
- Oracle path: `repros/canonical/sum_sum_sum_2334df16a35d/oracle_scatter_reduce.py`
- Measured oracle: _pending GPU measurement_
- Shape label: `timm_beit_base_patch16_224_train_001_68372476`

## Gap Diagnosis

Classification: **SCATTER_REDUCE**

This oracle computes the BEiT average-pool backward `slice_scatter` as a structured token scatter-reduce, deriving the `[768]` addmm-gradient reduction and `[768]` gamma-style reduction directly from the pre-scatter row gradient while materializing only the required `[768, 25216]` transposed side output. Inductor currently materializes the full `[128, 197, 768]` zero-prefix `slice_scatter` result and schedules the two reductions plus returned permute as generic consumers.

**Root cause**: Inductor's scheduler/codegen does not model `unsqueeze`/`expand`/`div` feeding `slice_scatter` as a structured scatter-reduce with a zero prefix, broadcasted patch tokens, materialized side stores, and reducer epilogues.

**Fix**: Add a structured average-pool-backward `slice_scatter` lowering that accumulates reductions from the source row-gradient tile while emitting any required side-output stores.

## Oracle Implementation

- **PyTorch implementation**: `oracle_scatter_reduce` -- computes LayerNorm row gradient, divides by PATCH_TOKENS for the patch gradient, then derives:
  - `out_addmm [768]`: weighted sum of patch tokens from `arg306_1`.
  - `transposed_side [768, 25216]`: materialized scatter with zero prefix, transposed.
  - `out_scaled [768]`: sum over flat materialized output.
- **Key insight**: The `[128, 197, 768]` slice_scatter is structured: position 0 is always zero, positions 1..196 are a broadcast of the `[128, 768]` patch gradient. The reductions can be computed directly from the `[128, 768]` row gradient without materializing the full 3D tensor first.
- **Total bytes**: 155,726,336 (6 kernels in compiled version)
- **Shape**: N=128, PATCH_TOKENS=196, TOKENS_WITH_PREFIX=197, D=768

## Inductor Closure Path

- Implementation track: Structured slice_scatter-reduce template for ViT average-pool backward.
- Candidate hook: Detect `layernorm_grad -> div(patch_tokens) -> slice_scatter(zero_prefix) -> [reduction, permute, reduction]` in post-grad graph; compute reductions from the pre-scatter row gradient tile and emit the transposed side store in one pass.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, and oracle floor.
- Gating policy: gate on slice_scatter with zero prefix + dual reduction + permute pattern.

## Done Criteria

- Oracle correctness verified on GPU (--check passes).
- Oracle timing measured and appended to measured_oracle_floors.csv.
- Inductor path either reaches the oracle floor or has a gated implementation plan.
