# sum_sum_sum_5f0c557e2169

## Compile: 12.64us, Oracle: 10.62us, Gap: 1.19x

## Diagnosis: SCATTER_REDUCE

## Root cause

The repro (from DeiT tiny train) computes a layer-norm backward through a `select_scatter` that fills a [128,197,192] tensor with zeros and writes only token 0's [128,192] slice from `mm`. All downstream operations (mul, sum over channels, mul by div, sub, permute, etc.) operate on this mostly-zero tensor, wasting computation on 196/197 rows that are guaranteed zero.

The oracle recognizes this sparsity structure and:
1. Computes the channel reductions (`sum_dim_int_list_2`, `sum_dim_int_list_3`) directly from the [128,192] source (token 0 only), since the other 196 rows contribute nothing
2. Computes the non-sparse path (`mul_tensor_4 -> reshape -> permute + sum`) efficiently by recognizing that the select_scatter feeding into `mul * primals_150 * 192 - sum - mul*div*sum - sub - sub` yields zeros everywhere except token 0

Inductor generates 3 kernels that operate on the full [128,197,192] = 4.7M element tensor, performing all reductions and pointwise ops on the dense materialization.

## Kernel count

- Inductor: 3 kernels (operating on full [128,197,192] tensors)
- Oracle: fewer kernels, operating only on the non-zero [128,192] slice for the sparse reductions

## Config exploration

| Config | Time (us) |
|--------|--------:|
| combo_kernels + CDT | 12.64 |

Config changes cannot help because the gap is structural: Inductor materializes the full [128,197,192] scatter and computes dense reductions, while the oracle exploits sparsity.

## Note on select_scatter_sparsity pass

The `select_scatter_sparsity` pass (added in commit 94ad050dee0) was designed to handle exactly this pattern. However, it had a bug (topological ordering violation from inserting replacement nodes at the wrong graph position) that caused compilation to fail on this repro. The bug has been fixed: the pass now inserts nodes before `materialize_node` instead of before `output_node`, and uses `delete_user_cb` to avoid self-replacement cycles.

With the fix applied, compilation succeeds and the pass partially optimizes the sparse reductions. The remaining 1.19x gap suggests the pass could be enhanced to:
1. Also rewrite the non-sparse outputs (permute path) more aggressively
2. Better handle the multi-consumer pattern where select_scatter feeds both sparse-dim reductions AND channel-dim reductions AND a materialized output

## Fix path

- **File**: `/tmp/pytorch-work/torch/_inductor/fx_passes/select_scatter_sparsity.py`
- **Status**: Bug fixed, pass now fires. Further optimization potential exists.
- **Enhancement needed**: The pass currently only rewrites reductions that reduce over the sparse dim. For this repro, the `permute_default` output and `reshape_default_1` output still materialize through the full [128,197,192] tensor.

## Models affected

- timm_deit_tiny_patch16_224.fb_in1k_train

## Status: partial_fix (pass fires after bug fix, remaining gap from incomplete sparse propagation)
