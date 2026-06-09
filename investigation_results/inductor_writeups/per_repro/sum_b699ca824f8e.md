# sum_b699ca824f8e


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 30.27 us
- Ratio: N/A

## Queue Position

- Rank: 673
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `codex`
- Closure status: `implemented_unmeasured`
- Oracle status: `oracle_implemented_needs_measurement`

## Current Gap

- Best compile: `29.696 us`
- Memcopy SOL: `13.376 us`
- Launch-adjusted SOL gap: `1.170x`
- Oracle path: `repros/canonical/sum_b699ca824f8e/oracle_avgpool_backward_reduce.py`
- Measured oracle: _pending GPU measurement_
- Shape label: `timm_nfnet_l0_train_369f714a`

## Gap Diagnosis

Classification: **SCATTER_REDUCE**

This oracle computes the adaptive-average-pool backward broadcast and SiLU-gradient channel sum as a direct per-channel reduction from the `[128, 2304]` pooled gradient and `[128, 2304, 7, 7]` activation tensor. Inductor currently lowers the `full -> as_strided_scatter -> as_strided -> expand -> div -> pointwise -> sum([0, 2, 3])` chain as ordinary tensor producers and consumers.

**Root cause**: Inductor's scatter/view scheduler does not recognize a zero-fill structured scatter/expand whose only real consumer is a channel reduction, and cannot lower that pattern to an output-centric reduction.

**Fix**: Add an FX/post-grad rewrite for zero-fill structured scatter/expand feeding pointwise channel reductions and lower it to a direct channel-reduction template.

## Oracle Implementation

- **Triton kernel**: `_avgpool_backward_reduce_kernel` -- fuses sigmoid/SiLU-grad computation with spatial reduction, output-centric with atomic_add accumulation across N-tiles.
- **Torch fallback**: `torch_direct_avgpool_backward_reduce` -- computes sigmoid, SiLU-backward, spatial sum, and channel product-sum in PyTorch ops.
- **Key insight**: Avoids materializing the full `[128, 2304, 7, 7]` zero-fill `as_strided_scatter` + expand intermediate. The oracle reads `mm[128,2304]` and `conv[128,2304,7,7]` directly and produces a `[2304]` reduction.
- **Total bytes**: 58,991,616 (4 kernels in compiled version)

## Inductor Closure Path

- Implementation track: Structured scatter-reduce template for zero-fill expand + channel reduction patterns.
- Candidate hook: Detect `full(0) -> as_strided_scatter -> as_strided -> expand -> div -> pointwise -> sum` in post-grad FX graph; rewrite to a direct channel-reduction kernel that reads from the source tile.
- Benchmark policy: compare default, `coordinate_descent_tuning=True`, and the oracle floor; report excess_us.
- Gating policy: gate on pattern predicate (zero-fill structured scatter/expand feeding channel reduction).

## Done Criteria

- Oracle correctness verified on GPU (--check passes).
- Oracle timing measured and appended to measured_oracle_floors.csv.
- Inductor path either reaches the oracle floor or has a gated implementation plan.
