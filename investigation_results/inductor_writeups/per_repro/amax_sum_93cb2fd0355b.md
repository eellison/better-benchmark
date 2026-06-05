# amax_sum_93cb2fd0355b

## Summary
- **Model**: torchbench_hf_Longformer_train_002
- **Classification**: NEW_PATTERN
- **Ratio**: 2.039x (oracle 97.89us vs compile 199.55us)
- **Status**: Genuine gap, requires new pattern/scheduler enhancement

## Root Cause

The oracle fuses the complete Longformer sliding-window attention assembly -- diagonal-skew indexing, band assembly from `bmm_22` via the `constant_pad_nd/view/slice_scatter/permute` chain, two conditional mask substitutions (`where`), position bias addition, row-wise online softmax, dropout with inductor RNG seeds, and the final skewed output layout -- into a single Triton row kernel (24,576 rows) plus a zero-fill of the output storage.

Inductor cannot perform this fusion because:
1. The slice_scatter/select_scatter/constant_pad_nd chain implements Longformer's diagonal-skew indexing pattern, which Inductor materializes as 5 separate pointwise kernels
2. The softmax reduction kernel cannot fuse backward through the materialized slice_scatter intermediates
3. The output layout (strided [96, 768, 256] with stride [197120, 1, 769]) requires careful zero-padded storage that Inductor handles with a separate zero-fill kernel

## Kernel Count
- **Oracle**: 2 kernels (zero-fill + fused longformer softmax/dropout)
- **Inductor**: 8 kernels:
  - `triton_poi_fused_0`: seed/RNG setup
  - `triton_poi_fused_1`: initial band assembly
  - `triton_poi_fused_constant_pad_nd_copy_permute_slice_view_2`: skew computation part 1
  - `triton_poi_fused_constant_pad_nd_copy_permute_slice_view_3`: skew computation part 2
  - `triton_poi_fused_constant_pad_nd_copy_permute_select_slice_view_4`: final skew assembly
  - `triton_red_fused_add_clone_copy_permute_prepare_softmax_online_slice_view_where_5`: softmax reduction
  - `triton_poi_fused_clone_constant_pad_nd_div_exp_gt_mul_permute_sub_view_where_6`: softmax epilogue + dropout
  - `triton_poi_fused_clone_constant_pad_nd_div_exp_gt_mul_permute_sub_view_where_7`: zero-fill padding region

## Config Exploration

| Config | Ratio | Status |
|--------|-------|--------|
| combo_kernels + CDT | 2.039 | Gap |
| combo_kernels + CDT + multi_kernel=3 | 2.233 | Gap (worse) |
| combo_kernels + CDT + multi_kernel=3 + max_autotune | 2.209 | Gap |

No config combination closes this gap.

## Intermediate Buffers Materialized

1. `buf0`: (2, 1024, 12, 513) f32 -- 50.3MB (RNG fill)
2. `buf1`: (24, 256, 513) f32 -- 12.5MB (initial band slice)
3. `buf2`: (24, 3, 256, 513) f32 -- 37.5MB (skew stage 1)
4. `buf3`: (24, 4, 256, 513) f32 -- 50.0MB (skew stage 2)
5. `buf4`: (24, 4, 256, 513) f32 -- 50.0MB (final assembly)
6. `buf6/buf7`: (2, 1024, 12, 1) f32 -- softmax stats

Total extra memory traffic: ~200MB of intermediate writes that the oracle avoids by computing all values inline per row.

## Design Doc

### Why It Cannot Be Fixed Today

The Longformer attention pattern uses a **diagonal-skew indexing scheme** where the attention scores for position `pos` attending to a sliding window are packed into a diagonal layout via `constant_pad_nd + view`. This creates a complex dependency between the source tensor layout and the per-row attention window, which Inductor's scheduler cannot resolve through fusion alone because:

1. **slice_scatter forces materialization**: Each `slice_scatter` in the chain creates a new tensor that overwrites a subset of its input. Inductor must materialize these because subsequent reads depend on the full tensor including both scattered and non-scattered regions.

2. **Non-affine indexing**: The diagonal-skew pattern maps `(row, col)` to `row * 513 + col` then re-interprets as `(src_row, src_col) = divmod(flat, 512)`. This is not expressible as an affine index transformation, so Inductor's fusion system (which assumes affine or simple indirect indexing) cannot fold it.

3. **Cross-chunk dependencies**: Chunks 0-3 of the window attention read from different regions of the skewed tensor, with special cases at boundaries (first/last chunk). The oracle handles this with explicit conditional loads; Inductor would need a dedicated lowering.

### What Enhancement Is Needed

A **Longformer sliding-window attention pattern recognizer** in `torch/_inductor/fx_passes/` that:
1. Detects the `view -> permute -> constant_pad_nd -> view -> slice -> slice_scatter` chain that implements diagonal-skew
2. Rewrites it to direct indexed loads from the source BMM output
3. Fuses the resulting row-wise computation (bias + mask + softmax + dropout) into a single persistent kernel

### Affected Files
- `torch/_inductor/fx_passes/post_grad.py` -- pattern registration
- `torch/_inductor/scheduler.py` -- slice_scatter fusion blocking
- `torch/_inductor/ir.py` -- realize_hint on slice_scatter intermediates

### Affected Repro Count
This pattern is specific to Longformer-family models. Related repros: `amax_sum_67d7c2666a5c` (same NEW_PATTERN classification for Longformer).
