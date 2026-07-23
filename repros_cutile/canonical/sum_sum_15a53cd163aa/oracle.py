"""cuTile port of sum_sum_15a53cd163aa: ALBERT MLM softmax backward.

Three cuTile kernels, mirroring Triton:
  1. materialize: writes out[M, N] with softmax-derivative + target one-hot.
  2. partial_sum: reduces out over row-blocks (per column block).
  3. final_sum: reduces partial rows to a single per-column bf16-rounded sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 4096
N = 30000

# Materialize block sizes: 1 row per program, 512 cols per tile.
MAT_ROW_BLOCK = 1
MAT_COL_BLOCK = 512

# Partial-sum block sizes: 128 rows per group → 32 groups over M=4096.
PSUM_ROW_BLOCK = 128
PSUM_COL_BLOCK = 64

# Final-sum block sizes: reduce over 32 groups, 64 cols per tile.
FSUM_GROUP_BLOCK = 32   # NUM_GROUPS = 32, power of two.
FSUM_COL_BLOCK = 64


@ct.kernel
def _materialize_kernel(
    numerator_ptr,     # f32 (1,)
    denominator_ptr,   # f32 (1,)
    labels_ptr,        # i64 (M,)
    logits_ptr,        # bf16 (M, N)
    row_max_ptr,       # f32 (M,)
    row_logsum_ptr,    # f32 (M,)
    residual_ptr,      # bf16 (M, N)
    out_ptr,           # bf16 (M, N)
    N_SIZE: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)

    numerator = ct.load(numerator_ptr, index=(0,), shape=(1,))
    denominator = ct.load(denominator_ptr, index=(0,), shape=(1,))
    scale = ct.astype(numerator, ct.float32) / ct.astype(denominator, ct.float32)
    neg_scale = ct.astype(ct.astype(-scale, ct.bfloat16), ct.float32)

    label = ct.load(labels_ptr, index=(row,), shape=(1,))
    minus_100 = ct.full((1,), -100, dtype=ct.int64)
    zero_i = ct.zeros((1,), dtype=ct.int64)
    n_i = ct.full((1,), N_SIZE, dtype=ct.int64)
    valid = label != minus_100
    in_range = valid & (label >= zero_i) & (label < n_i)
    zero_f = ct.zeros((1,), dtype=ct.float32)
    row_sum = ct.where(in_range, neg_scale, zero_f)  # (1,)

    logits = ct.load(
        logits_ptr, index=(row, col_block), shape=(1, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, col_block), shape=(1, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row_max = ct.load(row_max_ptr, index=(row,), shape=(1,))
    row_logsum = ct.load(row_logsum_ptr, index=(row,), shape=(1,))

    logits_f = ct.astype(logits, ct.float32)
    centered_bf = ct.astype(
        logits_f - ct.reshape(row_max, (1, 1)) - ct.reshape(row_logsum, (1, 1)),
        ct.bfloat16,
    )
    probs = ct.exp(ct.astype(centered_bf, ct.float32))

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_offset = col_block * BLOCK_C
    global_col = col_offset + cols
    global_col_i64 = ct.astype(global_col, ct.int64)
    global_col_2d = ct.reshape(global_col_i64, (1, BLOCK_C))
    label_2d = ct.reshape(label, (1, 1))
    is_target = global_col_2d == label_2d
    in_range_2d = ct.reshape(in_range, (1, 1))
    is_target_and_valid = is_target & in_range_2d
    row_sum_2d = ct.reshape(row_sum, (1, 1))
    zero_2d = ct.zeros((1, BLOCK_C), dtype=ct.float32)
    target_grad = ct.where(is_target_and_valid, row_sum_2d, zero_2d)

    grad_bf = ct.astype(target_grad - probs * row_sum_2d, ct.bfloat16)
    out_bf = ct.astype(
        ct.astype(residual, ct.float32) + ct.astype(grad_bf, ct.float32),
        ct.bfloat16,
    )
    col_mask_2d = ct.reshape(global_col < N_SIZE, (1, BLOCK_C))
    zero_bf_2d = ct.zeros((1, BLOCK_C), dtype=ct.bfloat16)
    ct.store(out_ptr, index=(row, col_block), tile=ct.where(col_mask_2d, out_bf, zero_bf_2d))


@ct.kernel
def _partial_sum_kernel(
    out_ptr,           # bf16 (M, N)
    partial_ptr,       # f32 (NUM_GROUPS, N)
    N_SIZE: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_group = ct.bid(0)
    col_block = ct.bid(1)

    values = ct.load(
        out_ptr, index=(row_group, col_block),
        shape=(ROW_BLOCK, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    values_f = ct.astype(values, ct.float32)
    # Column-mask OOB cols to zero.
    cols = ct.arange(BLOCK_C, dtype=ct.int32) + col_block * BLOCK_C
    col_active = cols < N_SIZE
    col_mask_2d = ct.reshape(col_active, (1, BLOCK_C))
    zero_2d = ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.float32)
    values_masked = ct.where(col_mask_2d, values_f, zero_2d)
    partial = ct.sum(values_masked, axis=0)  # (BLOCK_C,)
    partial_masked = ct.where(col_active, partial, ct.zeros((BLOCK_C,), dtype=ct.float32))
    # Write to partial[row_group, col_block*BLOCK_C : ...]
    row_g = ct.full((BLOCK_C,), row_group, dtype=ct.int32)
    ct.scatter(partial_ptr, (row_g, cols), partial_masked, mask=col_active)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,       # f32 (NUM_GROUPS, N)
    sum_ptr,           # f32 (N,)
    N_SIZE: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
    GROUP_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    values = ct.load(
        partial_ptr, index=(0, col_block), shape=(GROUP_BLOCK, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    groups = ct.arange(GROUP_BLOCK, dtype=ct.int32)
    group_active = groups < NUM_GROUPS
    group_mask_2d = ct.reshape(group_active, (GROUP_BLOCK, 1))
    cols = ct.arange(BLOCK_C, dtype=ct.int32) + col_block * BLOCK_C
    col_active = cols < N_SIZE
    col_mask_2d = ct.reshape(col_active, (1, BLOCK_C))
    active = group_mask_2d & col_mask_2d
    zero_2d = ct.zeros((GROUP_BLOCK, BLOCK_C), dtype=ct.float32)
    values_masked = ct.where(active, values, zero_2d)
    total = ct.sum(values_masked, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    rounded_masked = ct.where(col_active, rounded, ct.zeros((BLOCK_C,), dtype=ct.float32))
    ct.scatter(sum_ptr, (cols,), rounded_masked, mask=col_active)


@oracle_impl(hardware="B200", point="a93d5c9d")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, *_ = inputs
    device = arg3_1.device

    out = torch.empty_strided((M, N), (N, 1), device=device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((N,), (1,), device=device, dtype=torch.float32)

    logits_2d = arg3_1.reshape(M, N)
    residual_2d = arg6_1.reshape(M, N)
    labels_flat = arg2_1.reshape(M)
    row_max_flat = arg4_1.reshape(M)
    row_logsum_flat = arg5_1.reshape(M)

    num_groups = ct.cdiv(M, PSUM_ROW_BLOCK)  # 32
    partial = torch.zeros((num_groups, N), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (M, ct.cdiv(N, MAT_COL_BLOCK), 1), _materialize_kernel,
        (arg0_1.reshape(1), arg1_1.reshape(1), labels_flat, logits_2d,
         row_max_flat, row_logsum_flat, residual_2d, out,
         N, MAT_COL_BLOCK),
    )
    ct.launch(
        stream, (num_groups, ct.cdiv(N, PSUM_COL_BLOCK), 1), _partial_sum_kernel,
        (out, partial, N, PSUM_ROW_BLOCK, PSUM_COL_BLOCK),
    )
    ct.launch(
        stream, (ct.cdiv(N, FSUM_COL_BLOCK), 1, 1), _final_sum_kernel,
        (partial, sum_out, N, num_groups, FSUM_GROUP_BLOCK, FSUM_COL_BLOCK),
    )
    return out, out.permute(1, 0), sum_out
