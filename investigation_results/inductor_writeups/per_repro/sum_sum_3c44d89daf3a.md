# sum_sum_3c44d89daf3a

## Status

- Family: `structured_scatter_reduce`
- Closure status: `open_gap`
- Artifact: `repros/canonical/sum_sum_3c44d89daf3a/oracle_structured_scatter_reduce.py`
- Classification: `SCATTER_REDUCE`
- Label: `timm_mobilenetv3_large_100_train_001`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py`:
- `f32[512, 960, 1, 1]` pooled gradient
- `f32[512, 960, 7, 7]` activation
- `f32[1, 960, 1, 1]` mean, `f32[1, 960, 1, 1]` invstd
- `f32[960]` weight, `f32[960]` bias
- `f32[]` scalar zero
- One shape parameter

Returns: `f32[512, 960, 7, 7]` BN input gradient, `f32[960]` channel reduction.

## Timings

- Oracle: 91.81 us
- torch.compile (coordinate_descent_tuning=True, combo_kernels=True): 116.64 us
- Ratio: 1.27x
- multi_kernel=3: ~119 us (no improvement)

## Kernel Count

- **Oracle**: 4 kernels (per-sample reduce, finalize stats, epilogue, or single onepass for small N)
- **Inductor**: 1 kernel (triton_red_fused_add_div_expand_le_lt_mul_squeeze_sub_sum_unsqueeze_where_0)

## Root Cause

Despite Inductor already fusing everything into a single kernel, the oracle's
multi-kernel approach with per-channel tiling is faster because of memory access
pattern optimization.

The oracle decomposes the work into:
1. Per-sample partial reductions (accumulate grad_sum and centered_grad_sum per batch)
2. Finalize: sum partials across batch to get channel statistics
3. Epilogue: write full-tensor output using precomputed channel statistics

This 3-pass approach benefits from:
- Each pass has better cache locality (reading N*HW per channel vs channel per N*HW)
- The reduction passes use small BLOCK_HW tiles with good register pressure
- Channel statistics are computed once and broadcast in the epilogue

Inductor's single-kernel approach iterates per-channel with a large R0_BLOCK,
which forces each threadblock to load all N*HW=25088 elements per channel.
The oracle's split approach amortizes the two-pass overhead by having better
data locality in each phase.

The fundamental issue is that for this shape (N=512, C=960, HW=49), the optimal
strategy is a split reduction: first reduce over HW per (N,C), then reduce over N
per C, then write the full output. Inductor's single-kernel approach with one
reduction per channel forces large R0 blocks.

## Config Exploration

| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo+cdt (default harness) | 116.64 | 1 kernel, large reduction |
| combo+cdt+multi_kernel=3 | ~119 | no improvement |

## Design Doc

**Why Inductor cannot do this today**: Inductor correctly fuses the entire pattern
into one kernel. However, its reduction strategy (iterating N*HW per channel)
creates suboptimal memory access patterns for this shape. The `choices.py`
heuristic picks a persistent/looped reduction that iterates all 25088 elements
per channel, while the oracle splits into HW-local tiles first.

**What enhancement is needed**: The reduction strategy in `torch/_inductor/choices.py`
should consider a hierarchical/split reduction for BN-backward-shaped patterns where:
- The reduction domain is N*HW but the output writes both a per-channel scalar AND
  a full [N,C,H,W] tensor
- Splitting into per-sample HW reductions + cross-N finalization + epilogue
  gives better cache behavior

This is related to the `split_reductions` mechanism but specifically for patterns
where the epilogue depends on the reduction result and writes a full-size output.
The scheduler could detect this "reduce-then-broadcast-write" pattern and emit
the 3-phase structure.

**Affected files**:
- `torch/_inductor/choices.py` (split reduction heuristic)
- `torch/_inductor/scheduler.py` (detect reduce-then-epilogue patterns)
- Possibly `torch/_inductor/codegen/triton.py` for the multi-phase codegen

**Similar repros**: Other `structured_scatter_reduce` family repros with similar
avgpool-backward + BN-backward patterns (e.g., `sum_sum_3219a09ab96a`,
`sum_sum_145a8857ebf2`, `sum_sum_1a663e225990`).

## Validation

- `oracle_structured_scatter_reduce.py --check`: PASS
- `oracle_structured_scatter_reduce.py --bench`: ratio 1.27x, status GOOD
