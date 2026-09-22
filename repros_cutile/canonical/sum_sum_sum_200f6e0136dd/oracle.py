"""cuTile port of sum_sum_sum_200f6e0136dd: BERT layer-norm-backward /
dropout embedding-gradient tuple with atomic scatter accumulation.

Structure:
  - Torch: zero-init the segment [3,768] and vocab [20005,768] outputs.
  - cuTile row-tile kernel: computes the producer f32 [rows, C] (3-bf16 sum,
    layer-norm-backward chain, dropout scale), scatters via ct.atomic_add
    into segment_out and vocab_out with row_idx masked to OOB (-1) when
    invalid, and emits per-group column partials for the two side-output
    reductions.
  - cuTile finalize kernel: reduces the [num_groups, 2, C] partials to two
    contiguous [768] column sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 2048
CHANNELS = 768
SEGMENT_ROWS = 3
VOCAB_ROWS = 20005
EPS = 1.0e-6
ROW_BACKWARD_SCALE = 0.002607561929595828
RETURN_DROPOUT_SCALE = 1.1111111111111112
INV_CHANNELS = 1.0 / 768.0


@ct.kernel
def _row_scatter_kernel(
    arg0_ptr,       # bf16 [rows, C]
    arg1_ptr,       # bf16 [rows, C]
    arg2_ptr,       # bf16 [rows, C]
    gamma_ptr,      # f32 [C]
    dy_ptr,         # f32 [rows, C]
    denom_base_ptr, # f32 [rows]
    residual_ptr,   # f32 [rows, C]
    full_scalar_ptr, # f32 [1]
    keep_ptr,       # b8 [rows, C]
    seg_idx_ptr,    # i64 [rows]
    vocab_idx_ptr,  # i64 [rows]
    segment_out_ptr, # f32 [SEGMENT_ROWS, C]
    vocab_out_ptr,   # f32 [VOCAB_ROWS, C]
    partials_ptr,    # f32 [num_groups, 2, C] flat
    ROWS_PER_GROUP: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    C_: ct.Constant[int],
    SEGMENT_ROWS_: ct.Constant[int],
    VOCAB_ROWS_: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
):
    group = ct.bid(0)
    # channel columns (padded to BLOCK_C, mask out of range)
    cols = ct.arange(BLOCK_C, dtype=ct.int64)
    c_mask = cols < C_

    gamma_1d = ct.load(gamma_ptr, index=(0,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    full_scalar_tile = ct.load(full_scalar_ptr, index=(0,), shape=(1,))

    acc_sum_x = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_sum_x_dy_over_denom = ct.zeros((BLOCK_C,), dtype=ct.float32)

    for local_start in range(0, ROWS_PER_GROUP, BLOCK_R):
        # Absolute row indices for this micro-tile
        row_offsets = ct.arange(BLOCK_R, dtype=ct.int64) + (group * ROWS_PER_GROUP + local_start)
        row_mask = row_offsets < ROWS
        mask2d_r = ct.expand_dims(row_mask, axis=1)  # (BR, 1)
        mask2d_c = ct.expand_dims(c_mask, axis=0)  # (1, BC)
        mask = mask2d_r & mask2d_c

        base_row = group * ROWS_PER_GROUP + local_start
        # Load 2D tile [BLOCK_R, BLOCK_C] at (base_row, 0).
        a0_bf = ct.load(arg0_ptr, index=(base_row, 0), shape=(BLOCK_R, BLOCK_C),
                        padding_mode=ct.PaddingMode.ZERO)
        a1_bf = ct.load(arg1_ptr, index=(base_row, 0), shape=(BLOCK_R, BLOCK_C),
                        padding_mode=ct.PaddingMode.ZERO)
        a2_bf = ct.load(arg2_ptr, index=(base_row, 0), shape=(BLOCK_R, BLOCK_C),
                        padding_mode=ct.PaddingMode.ZERO)
        a0 = ct.astype(a0_bf, ct.float32)
        a1 = ct.astype(a1_bf, ct.float32)
        a2 = ct.astype(a2_bf, ct.float32)
        x = a0 + a1 + a2

        dy = ct.load(dy_ptr, index=(base_row, 0), shape=(BLOCK_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
        residual = ct.load(residual_ptr, index=(base_row, 0),
                           shape=(BLOCK_R, BLOCK_C),
                           padding_mode=ct.PaddingMode.ZERO)
        denom_base_1d = ct.load(denom_base_ptr, index=(base_row,),
                                shape=(BLOCK_R,),
                                padding_mode=ct.PaddingMode.ZERO)
        eps_bn = ct.full((BLOCK_R,), EPS, dtype=ct.float32)
        denom_1d = denom_base_1d + eps_bn
        denom_2d = ct.expand_dims(denom_1d, axis=1)  # (BR, 1)

        gamma_2d = ct.expand_dims(gamma_1d, axis=0)  # (1, BC)
        gamma_dy = gamma_2d * dy
        gamma_dy_over_denom = gamma_dy / denom_2d
        gamma_dy_over_denom2 = gamma_dy_over_denom / denom_2d
        sum2_terms = (0.0 - x) * gamma_dy_over_denom2
        zero_2d = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=ct.float32)
        row_sum2 = ct.sum(ct.where(mask, sum2_terms, zero_2d), axis=1)

        x_over_denom = x / denom_2d
        x_gamma_over_denom = x_over_denom * gamma_2d
        x_dy_over_denom = x_over_denom * dy

        row_sum4 = 0.0 - ct.sum(
            ct.where(mask, x_gamma_over_denom, zero_2d), axis=1)
        denom_twice = denom_base_1d * 2.0
        div3 = row_sum2 / denom_twice
        # full_scalar broadcast to (BR,)
        full_bn = ct.broadcast_to(full_scalar_tile, (BLOCK_R,))
        where_val = ct.where(denom_base_1d == 0.0, full_bn, div3)
        row_coef = where_val * ROW_BACKWARD_SCALE

        add3 = residual + x_gamma_over_denom
        row_coef_2d = ct.expand_dims(row_coef, axis=1)
        add4 = add3 + row_coef_2d * dy
        div4_1d = row_sum4 * INV_CHANNELS
        div4_2d = ct.expand_dims(div4_1d, axis=1)
        add5 = add4 + div4_2d

        keep = ct.load(keep_ptr, index=(base_row, 0),
                       shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
        keep_f = ct.astype(keep, ct.float32)
        keep_scale = keep_f * RETURN_DROPOUT_SCALE
        producer = add5 * keep_scale

        # Column-partial accumulators (masked).
        producer_masked = ct.where(mask, producer, zero_2d)
        acc_sum_x = acc_sum_x + ct.sum(ct.where(mask, x, zero_2d), axis=0)
        acc_sum_x_dy_over_denom = acc_sum_x_dy_over_denom + ct.sum(
            ct.where(mask, x_dy_over_denom, zero_2d), axis=0)

        # Segment scatter: rows [3,768], indexed by seg_idx[row].
        seg_raw = ct.load(seg_idx_ptr, index=(base_row,),
                          shape=(BLOCK_R,),
                          padding_mode=ct.PaddingMode.ZERO)
        # segment_valid = (raw>=0) & (raw<3) & (raw!=0) & row_mask
        seg_valid = (seg_raw >= 0) & (seg_raw < SEGMENT_ROWS_) & (seg_raw != 0) & row_mask
        # Row idx = raw if valid else -1 (OOB filter)
        neg_bn_i64 = ct.full((BLOCK_R,), -1, dtype=ct.int64)
        seg_row_final = ct.where(seg_valid, seg_raw, neg_bn_i64)
        # Column idx (masked to OOB when c_mask False)
        big_col = ct.full((BLOCK_C,), 10**9, dtype=ct.int64)
        col_final = ct.where(c_mask, cols, big_col)
        seg_row_2d = ct.expand_dims(seg_row_final, axis=1)  # (BR, 1)
        col_2d = ct.expand_dims(col_final, axis=0)  # (1, BC)
        # atomic_add broadcast indices
        ct.atomic_add(segment_out_ptr, (seg_row_2d, col_2d), producer_masked)

        vocab_raw = ct.load(vocab_idx_ptr, index=(base_row,),
                            shape=(BLOCK_R,),
                            padding_mode=ct.PaddingMode.ZERO)
        vocab_valid = (vocab_raw >= 0) & (vocab_raw < VOCAB_ROWS_) & (vocab_raw != 0) & row_mask
        vocab_row_final = ct.where(vocab_valid, vocab_raw, neg_bn_i64)
        vocab_row_2d = ct.expand_dims(vocab_row_final, axis=1)
        ct.atomic_add(vocab_out_ptr, (vocab_row_2d, col_2d), producer_masked)

    # Store partials: layout is [num_groups, 2, C]. Column-only via 1D array.
    partial_base_1d = ct.arange(BLOCK_C, dtype=ct.int64) + (group * 2 * C_)
    partial_base_2 = partial_base_1d + C_
    big_off_1d = ct.full((BLOCK_C,), 10**9, dtype=ct.int64)
    part1_idx = ct.where(c_mask, partial_base_1d, big_off_1d)
    part2_idx = ct.where(c_mask, partial_base_2, big_off_1d)
    ct.scatter(partials_ptr, part1_idx, acc_sum_x)
    ct.scatter(partials_ptr, part2_idx, acc_sum_x_dy_over_denom)


@ct.kernel
def _finalize_partials_kernel(
    partials_ptr,   # f32 [num_groups, 2, C] flat
    sum_x_ptr,      # f32 [C]
    sum_x_dy_over_denom_ptr, # f32 [C]
    NUM_GROUPS: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_G: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    cols = ct.arange(BLOCK_C, dtype=ct.int64) + (col_block * BLOCK_C)
    c_mask = cols < C_

    # Load all group partials at once for these cols: shape (BLOCK_G, BLOCK_C).
    groups = ct.arange(BLOCK_G, dtype=ct.int64)
    g_mask = groups < NUM_GROUPS
    mask2d = ct.expand_dims(g_mask, axis=1) & ct.expand_dims(c_mask, axis=0)

    # Indices for partials_ptr: flat[group * 2*C + col]
    groups_2d = ct.expand_dims(groups, axis=1) * (2 * C_)
    cols_2d = ct.expand_dims(cols, axis=0)
    idx_sum_x = groups_2d + cols_2d
    idx_sum_dy = groups_2d + cols_2d + C_

    zero2d = ct.zeros((BLOCK_G, BLOCK_C), dtype=ct.float32)

    # Use scatter-style OOB filter for masked reads: gather with mask.
    big2d = ct.full((BLOCK_G, BLOCK_C), 10**12, dtype=ct.int64)
    sx_idx_safe = ct.where(mask2d, idx_sum_x, big2d)
    sy_idx_safe = ct.where(mask2d, idx_sum_dy, big2d)
    sx = ct.gather(partials_ptr, sx_idx_safe)
    sy = ct.gather(partials_ptr, sy_idx_safe)
    sx = ct.where(mask2d, sx, zero2d)
    sy = ct.where(mask2d, sy, zero2d)
    total_sx = ct.sum(sx, axis=0)
    total_sy = ct.sum(sy, axis=0)

    # Store to sum_x/sum_dy at cols positions, masked
    zero1d_bc = ct.zeros((BLOCK_C,), dtype=ct.float32)
    big1d = ct.full((BLOCK_C,), 10**9, dtype=ct.int64)
    cols_safe = ct.where(c_mask, cols, big1d)
    ct.scatter(sum_x_ptr, cols_safe, total_sx)
    ct.scatter(sum_x_dy_over_denom_ptr, cols_safe, total_sy)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(
    hardware="B200",
    point="3d639bb6",
    ROWS_PER_GROUP=4,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=64,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        arg8_1, arg9_1, arg10_1,
        _shape_param_0, _shape_param_1, _shape_param_2,
        shape_sum0, shape_sum1,
        _shape_param_5,
        shape_segment, shape_vocab,
    ) = inputs

    device = arg0_1.device
    sum0 = torch.zeros(_shape_tuple(shape_sum0), device=device, dtype=torch.float32)
    sum1 = torch.zeros(_shape_tuple(shape_sum1), device=device, dtype=torch.float32)
    segment_out = torch.zeros(_shape_tuple(shape_segment), device=device, dtype=torch.float32)
    vocab_out = torch.zeros(_shape_tuple(shape_vocab), device=device, dtype=torch.float32)

    n_rows = int(ROWS)
    channels = int(CHANNELS)
    num_groups = (n_rows + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP

    partials = torch.zeros((num_groups, 2, channels),
                           device=device, dtype=torch.float32)

    # Flatten inputs to their [rows, C] and [rows] forms:
    arg0_2d = arg0_1.contiguous().view(n_rows, channels)
    arg1_2d = arg1_1.contiguous().view(n_rows, channels)
    arg2_2d = arg2_1.contiguous().view(n_rows, channels)
    dy_2d = arg4_1.contiguous().view(n_rows, channels)
    denom_base_1d = arg5_1.contiguous().view(n_rows)
    residual_2d = arg6_1.contiguous().view(n_rows, channels)
    full_scalar_1 = arg7_1.contiguous().view(1)
    keep_2d = arg8_1.contiguous().view(n_rows, channels)
    seg_idx_1d = arg9_1.contiguous().view(n_rows).to(torch.int64)
    vocab_idx_1d = arg10_1.contiguous().view(n_rows).to(torch.int64)
    gamma_1d = arg3_1.contiguous()

    partials_flat = partials.view(-1)
    segment_flat = segment_out  # 2D array for cuTile Array with 2-D indices
    vocab_flat = vocab_out
    sum0_1d = sum0.view(channels)
    sum1_1d = sum1.view(channels)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, 1, 1),
        _row_scatter_kernel,
        (arg0_2d, arg1_2d, arg2_2d, gamma_1d, dy_2d, denom_base_1d,
         residual_2d, full_scalar_1, keep_2d, seg_idx_1d, vocab_idx_1d,
         segment_flat, vocab_flat, partials_flat,
         ROWS_PER_GROUP, BLOCK_R, BLOCK_C,
         channels, SEGMENT_ROWS, VOCAB_ROWS, num_groups),
    )
    block_g = _next_power_of_2(num_groups)
    n_col_blocks = (channels + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C
    ct.launch(
        stream,
        (n_col_blocks, 1, 1),
        _finalize_partials_kernel,
        (partials_flat, sum0_1d, sum1_1d,
         num_groups, channels, block_g, FINAL_BLOCK_C),
    )

    return sum0, sum1, segment_out, vocab_out
