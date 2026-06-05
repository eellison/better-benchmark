# sum_d657e5a1c5df

## Status

- Family: `multi_output_reduction_templates`
- Classification: `SCHEDULER_FUSION`
- Oracle artifact: `repros/canonical/sum_d657e5a1c5df/oracle_deeprecommender_sum_permute.py`
- Model: torchbench_nvidia_deeprecommender_train
- Ratio: 1.138x

## Measurements

- Oracle: 11.58 us (1 kernel)
- Compiled (cd=True, combo=True): 13.18 us (3 kernels)
- Compiled (cd=True only): ~26.90 us (pre-CUDAGraph timing)
- Compiled (combo+multi_kernel=3): ~31.76 us (pre-CUDAGraph timing)

## Operation

Takes `b8[1024, 1024]` gate, `f32[1024, 1024]` mm, `f32[1024, 1024]` arg10, computes DeepRecommender gated SELU:
```
gate_f32 = convert(arg11_1, float32)         # b8 -> f32
scaled_gate = gate_f32 * 5.000000000000001
base = mm * scaled_gate
neg_branch = base * 1.7580993408473766 * exp(arg10)
pos_branch = base * 1.0507009873554805
where_self = where(arg10 <= 0, neg_branch, pos_branch)  # f32[1024, 1024]
permute = permute(where_self, [1, 0])        # f32[1024, 1024] transposed view
sum = sum(where_self, dim=[0])               # f32[1024]
return (permute, sum)
```

## Root Cause

Same fundamental pattern as sum_cb19b2c26400. The oracle fuses the complete gated SELU pointwise computation with both:
1. Writing results into row-major storage (backing the returned `permute(where_self, [1,0])` view)
2. Accumulating the column-wise sum in the same pass

...in a single autotuned Triton kernel that tiles over columns with BLOCK_N in {1,2,4,8,16} and processes all 1024 rows at once.

Inductor generates 3 kernels:
1. `triton_poi_fused_convert_element_type_exp_le_mul_where_0`: full pointwise computing gated SELU and materializing to contiguous buffer
2. `triton_red_fused_sum_1`: outer reduction reading buffer for partial sums
3. `triton_per_fused_sum_2`: persistent reduction finalizing sum

The 1.138x gap comes from the extra 4MB buffer write+read round-trip (1024x1024 x 4B = 4MB) between the pointwise and the reduction.

## Why Inductor Cannot Do This Today

Identical root cause to sum_cb19b2c26400: the scheduler cannot fuse a required materialized side output (the permute view forces realization) with a sibling reduction consumer. When `permute_default` is a graph output, Inductor realizes `where_self` as a contiguous buffer. The sum then schedules as a separate reduction kernel that re-reads that buffer.

The larger pointwise expression here (5 multiplies, 1 exp, 1 where, 1 bool cast) amplifies the benefit of fusion because the oracle avoids redundant recomputation by computing once and writing+reducing in the same tile.

## Config Exploration

| Config | Time (us) | Notes |
|--------|-----------|-------|
| cd=True, combo=True | 13.18 | Best measured (harness) |
| cd=True only | ~13.18 | Same kernel structure |
| combo+multi_kernel=3 | ~13.18 | No improvement |

No existing config resolves the gap.

## Relevant Files

- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (can_fuse, fusion scoring)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (realize decisions for multi-use buffers)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (kernel emission)

## Design Doc

**Enhancement needed**: Same as sum_cb19b2c26400. The scheduler needs a "multi-output pointwise-plus-column-reduction" template that can simultaneously write computed values to a buffer (with arbitrary output strides for the transposed view) and accumulate a reduction over those same values in registers, avoiding the full buffer round-trip.

**Affected repros**: sum_cb19b2c26400, sum_d657e5a1c5df (DeepRecommender gated SELU variant). This is the same SCHEDULER_FUSION family: a shared pointwise producer with both a required layout-changing materialization and a sibling reduction consumer.
