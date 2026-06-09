# sum_sum_2847f46e0c5a

## Status

- Family: `structured_pool_upsample_backward_reduce`
- Closure status: `open_gap`
- Artifact: `repros/canonical/sum_sum_2847f46e0c5a/oracle_structured_pool_upsample_backward_reduce.py`
- Classification: `SCATTER_REDUCE`
- Label: `torchbench_resnet152_train_001`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py`:
- `f32[32, 64, 56, 56]` grad_a (two of them, added)
- `i8[32, 64, 56, 56]` max-pool offsets
- `f32[32, 64, 112, 112]` activation
- `f32[1, 64, 1, 1]` mean, `f32[1, 64, 1, 1]` invstd
- `f32[64]` weight, `f32[64]` bias
- `f32[]` scalar zero
- Three shape parameters

Returns: `f32[32, 64, 112, 112]` BN input gradient, `f32[64]` channel reduction.

## Timings

- Oracle: 99.14 us
- torch.compile (coordinate_descent_tuning=True, combo_kernels=True): 180.03 us
- Ratio: 1.816x
- multi_kernel=3: ~181 us (no improvement)

## Kernel Count

- **Oracle**: 3 kernels (partial_bn_sums, finalize_bn_sums, bn_input_grad)
- **Inductor**: 3 kernels (fused_low_memory_max_pool_offsets_to_indices_add_full_scatter_add_view_0 poi, fused_1 poi, fused_2 red)

## Root Cause

The oracle exploits the structured max-pool-backward pattern where each source pixel
at `[n, c, h, w]` maps deterministically to a 2x2 tile at `[n, c, 2*h:2*h+2, 2*w:2*w+2]`
in the output. By knowing this structure, the oracle:
1. Never materializes the dense `[2048, 12544]` scatter buffer
2. Iterates over source-space `[N*56*56]` tiles, unrolling the 2x2 output positions
3. Accumulates channel reductions while traversing source space
4. Writes the full `[32, 64, 112, 112]` output directly from source-aligned tiles

Inductor instead:
1. Materializes a `full([2048, 12544], 0)` buffer (96 MB of zeros)
2. Computes max-pool offset indices via `_low_memory_max_pool_offsets_to_indices`
3. Performs `scatter_add` into the full buffer (random writes, low bandwidth utilization)
4. Reads the materialized buffer back for the BN-backward reduction + epilogue

The key cost is the 96 MB scatter buffer: writing zeros, performing scatter_add with
indirect indices, then reading it back. The oracle avoids this entirely by knowing
the scatter is a structured 2x2 upsample (stride-2 max-pool backward with center offsets).

## Config Exploration

| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo+cdt (default harness) | 180.03 | 3 kernels, scatter buffer |
| combo+cdt+multi_kernel=3 | ~181 | no improvement |

## Design Doc

**Why Inductor cannot do this today**: The lowering for `_low_memory_max_pool_offsets_to_indices`
+ `scatter_add` treats the scatter as a generic indirect-index operation. The scheduler
cannot recognize that the offsets encode a structured 2x upsample pattern (each source
pixel writes to exactly one known 2x2 location). The `scatter_add` is realized as a
buffer that blocks fusion with downstream consumers.

**What enhancement is needed**: A structured max-pool-backward scatter-reduce FX pass
or IR lowering that:
1. Detects `_low_memory_max_pool_offsets_to_indices` -> `scatter_add` -> reduction pattern
2. Recognizes stride-2 pool with 3x3 kernel/padding produces a 2x2 upsample structure
3. Replaces the scatter with a source-space iteration that writes outputs directly
4. Fuses the BN-backward channel reductions into the same traversal

**Affected files**:
- `torch/_inductor/lowering.py` (or a new FX pass in `fx_passes/`)
- `torch/_inductor/scheduler.py` (to not realize the scatter buffer when structure detected)

**Similar repros**: `sum_sum_1efe819604dc` (oracle_structured_pool_upsample_reduce),
`sum_sum_4bd81dea302d` (oracle_structured_pool_upsample_backward_reduce) - same pattern
family with different shapes.

## Validation

- `oracle_structured_pool_upsample_backward_reduce.py --check`: PASS
- `oracle_structured_pool_upsample_backward_reduce.py --bench`: ratio 1.816x, status GOOD
