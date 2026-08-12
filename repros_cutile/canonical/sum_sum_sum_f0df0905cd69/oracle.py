"""cuTile port of sum_sum_sum_f0df0905cd69: BERT embedding scatter-reduce.

Structure mirrors Triton's 3-kernel plan:
  1) _row_ln_backward_kernel: per row group, compute the LN-backward
     producer (add_2, dropped x, mul_6 grad), plus row_sum/row_dot
     reductions and column partials for sum_3 (x*arg7) and sum_4 (x).
     Also computes sum_5 partial (per (token, hidden)) for the pos scatter.
  2) _finalize_partials_kernel: sum column partials -> sum_3, sum_4.
  3) _scatter_kernels (2): reuse the existing atomic scatters for
     pos_out, segment_out, vocab_out.

Reductions in this port are done inside cuTile (matches Triton). All
scatters go to atomic_add via cuTile (matches Triton's atomic_add).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ           # 16384
HIDDEN = 768
HIDDEN_PAD = 1024
POSITION_ROWS = 512
SEGMENT_ROWS = 2
VOCAB_ROWS = 30522
DROP_SCALE = 1.1111111111111112

BLOCK_H = 128  # tile size for the scatter kernel; HIDDEN=768 -> 6 tiles
ROWS_PER_GROUP = 8
NUM_GROUPS = (ROWS + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _row_ln_backward_kernel(
    arg1_ptr,     # bf16 [ROWS, HIDDEN]  bf16_a (view of arg1 reshaped)
    arg2_ptr,     # f32  [ROWS, HIDDEN]  residual
    arg3_ptr,     # bf16 [ROWS, HIDDEN]  bf16_b
    arg4_ptr,     # bf16 [ROWS, HIDDEN]  bf16_c
    keep_ptr,     # f32  [ROWS, HIDDEN]  arg5 (dropout mask)
    weight_ptr,   # f32  [HIDDEN]        arg6
    normed_ptr,   # f32  [ROWS, HIDDEN]  arg7
    scale_ptr,    # f32  [ROWS]          arg8
    grad_out_ptr, # f32  [ROWS, HIDDEN]  mul_6
    sum5_out_ptr, # f32  [SEQ, HIDDEN]   sum_5 (per-token partial across batch)
    partial_x_normed_ptr, # f32 [NUM_GROUPS, HIDDEN]  sum_3 partial
    partial_x_ptr,        # f32 [NUM_GROUPS, HIDDEN]  sum_4 partial
    ROWS_PER_GROUP_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    HIDDEN_PAD_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    ROWS_: ct.Constant[int],
    DROP_SCALE_: ct.Constant[float],
):
    group = ct.bid(0)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PAD_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    cols = ct.arange(HIDDEN_PAD_, dtype=ct.int32)
    col_valid_1d = cols < HIDDEN_
    col_valid = ct.reshape(col_valid_1d, (1, HIDDEN_PAD_))
    zeros_2d = ct.full((1, HIDDEN_PAD_), 0.0, dtype=ct.float32)

    acc_x_normed = ct.zeros((HIDDEN_PAD_,), dtype=ct.float32)
    acc_x = ct.zeros((HIDDEN_PAD_,), dtype=ct.float32)

    for row_offset in ct.static_iter(range(ROWS_PER_GROUP_)):
        row = group * ROWS_PER_GROUP_ + row_offset

        # Load producer inputs.
        arg1 = ct.astype(
            ct.load(arg1_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        arg2 = ct.load(arg2_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
                       padding_mode=ct.PaddingMode.ZERO)
        arg3 = ct.astype(
            ct.load(arg3_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        arg4 = ct.astype(
            ct.load(arg4_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        add_2 = arg2 + arg1 + arg3 + arg4  # f32 [1, HIDDEN_PAD]

        keep = ct.astype(
            ct.load(keep_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        drop = keep * DROP_SCALE_
        x = add_2 * drop  # (1, HIDDEN_PAD)

        normed = ct.load(normed_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
                         padding_mode=ct.PaddingMode.ZERO)
        row_scale = ct.load(scale_ptr, index=(row,), shape=(1,),
                            padding_mode=ct.PaddingMode.ZERO)
        row_scale_2d = ct.reshape(row_scale, (1, 1))

        weight_2d = ct.reshape(weight, (1, HIDDEN_PAD_))
        weighted = x * weight_2d
        # Mask OOB cols before reducing.
        weighted_m = ct.where(col_valid, weighted, zeros_2d)
        row_sum = ct.sum(weighted_m, axis=1, keepdims=True)  # (1, 1)
        weighted_normed_m = ct.where(col_valid, weighted * normed, zeros_2d)
        row_dot = ct.sum(weighted_normed_m, axis=1, keepdims=True)  # (1, 1)
        centered = weighted * HIDDEN_ - row_sum - normed * row_dot
        grad = row_scale_2d * centered  # (1, HIDDEN_PAD)

        # Store only the first HIDDEN elements of grad — use scatter with mask.
        row_bcast = ct.full((1, HIDDEN_PAD_), row, dtype=ct.int32)
        cols_2d = ct.reshape(cols, (1, HIDDEN_PAD_))
        ct.scatter(grad_out_ptr, (row_bcast, cols_2d), grad, mask=col_valid)

        # Column partials — mask OOB before accumulating.
        x_1d = ct.reshape(ct.where(col_valid, x, zeros_2d), (HIDDEN_PAD_,))
        xn_1d = ct.reshape(ct.where(col_valid, x * normed, zeros_2d), (HIDDEN_PAD_,))
        acc_x_normed = acc_x_normed + xn_1d
        acc_x = acc_x + x_1d

        # sum_5 = sum(mul_6, dim=0) — per-token accumulation across batch.
        # Filter OOB cols by remapping them to negative index (check_bounds skips them).
        token = row - (row // SEQ_) * SEQ_  # row % SEQ
        token_bc = ct.full((1, HIDDEN_PAD_), token, dtype=ct.int32)
        neg_ones = ct.full((1, HIDDEN_PAD_), -1, dtype=ct.int32)
        cols_masked = ct.where(col_valid, cols_2d, neg_ones)
        ct.atomic_add(sum5_out_ptr, (token_bc, cols_masked), grad)

    ct.store(partial_x_normed_ptr, index=(group, 0),
             tile=ct.reshape(acc_x_normed, (1, HIDDEN_PAD_)))
    ct.store(partial_x_ptr, index=(group, 0),
             tile=ct.reshape(acc_x, (1, HIDDEN_PAD_)))


@ct.kernel
def _finalize_partials_kernel(
    partial_x_normed_ptr,   # f32 [NUM_GROUPS, HIDDEN_PAD]
    partial_x_ptr,          # f32 [NUM_GROUPS, HIDDEN_PAD]
    out_x_normed_ptr,       # f32 [HIDDEN_PAD]  (only first HIDDEN valid)
    out_x_ptr,              # f32 [HIDDEN_PAD]
    NUM_GROUPS_: ct.Constant[int],
    NUM_GROUPS_PAD: ct.Constant[int],
    HIDDEN_PAD_: ct.Constant[int],
):
    p_x_normed = ct.load(partial_x_normed_ptr, index=(0, 0),
                         shape=(NUM_GROUPS_PAD, HIDDEN_PAD_),
                         padding_mode=ct.PaddingMode.ZERO)
    p_x = ct.load(partial_x_ptr, index=(0, 0),
                  shape=(NUM_GROUPS_PAD, HIDDEN_PAD_),
                  padding_mode=ct.PaddingMode.ZERO)
    idx = ct.arange(NUM_GROUPS_PAD, dtype=ct.int32)
    valid = ct.reshape(idx < NUM_GROUPS_, (NUM_GROUPS_PAD, 1))
    zeros = ct.full((NUM_GROUPS_PAD, HIDDEN_PAD_), 0.0, dtype=ct.float32)
    p_x_normed_m = ct.where(valid, p_x_normed, zeros)
    p_x_m = ct.where(valid, p_x, zeros)

    s_x_normed = ct.sum(p_x_normed_m, axis=0)
    s_x = ct.sum(p_x_m, axis=0)
    ct.store(out_x_normed_ptr, index=(0,), tile=s_x_normed)
    ct.store(out_x_ptr, index=(0,), tile=s_x)


@ct.kernel
def _scatter_seg_voc_kernel(
    grad_ptr,             # f32 [ROWS, HIDDEN]
    segment_index_ptr,    # i64 [ROWS]
    vocab_index_ptr,      # i64 [ROWS]
    segment_out_ptr,      # f32 [SEGMENT_ROWS, HIDDEN]
    vocab_out_ptr,        # f32 [VOCAB_ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    h_tile = ct.bid(1)

    grad_v = ct.load(grad_ptr, index=(row, h_tile), shape=(1, BLOCK_H_),
                     padding_mode=ct.PaddingMode.ZERO)
    grad_v = ct.reshape(grad_v, (BLOCK_H_,))
    cols = ct.arange(BLOCK_H_, dtype=ct.int32) + h_tile * BLOCK_H_

    seg_row_scalar = ct.load(segment_index_ptr, index=(row,), shape=(1,),
                              padding_mode=ct.PaddingMode.ZERO)
    seg_row_bc = ct.broadcast_to(
        ct.astype(ct.reshape(seg_row_scalar, (1,)), ct.int32), (BLOCK_H_,)
    )
    ct.atomic_add(segment_out_ptr, (seg_row_bc, cols), grad_v)

    vocab_row_scalar = ct.load(vocab_index_ptr, index=(row,), shape=(1,),
                                padding_mode=ct.PaddingMode.ZERO)
    vocab_row_bc = ct.broadcast_to(
        ct.astype(ct.reshape(vocab_row_scalar, (1,)), ct.int32), (BLOCK_H_,)
    )
    # Skip row==0 for vocab (as in the original).
    vocab_row_masked = ct.where(vocab_row_bc == 0, ct.astype(-1, ct.int32),
                                vocab_row_bc)
    ct.atomic_add(vocab_out_ptr, (vocab_row_masked, cols), grad_v)


@ct.kernel
def _scatter_pos_kernel(
    sum_5_ptr,            # f32 [SEQ, HIDDEN]
    pos_index_ptr,        # i64 [SEQ]
    pos_out_ptr,          # f32 [POSITION_ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    token = ct.bid(0)
    h_tile = ct.bid(1)

    sum5_v = ct.load(sum_5_ptr, index=(token, h_tile), shape=(1, BLOCK_H_),
                      padding_mode=ct.PaddingMode.ZERO)
    sum5_v = ct.reshape(sum5_v, (BLOCK_H_,))
    cols = ct.arange(BLOCK_H_, dtype=ct.int32) + h_tile * BLOCK_H_

    pos_row_scalar = ct.load(pos_index_ptr, index=(token,), shape=(1,),
                               padding_mode=ct.PaddingMode.ZERO)
    pos_row_bc = ct.broadcast_to(
        ct.astype(ct.reshape(pos_row_scalar, (1,)), ct.int32), (BLOCK_H_,)
    )
    ct.atomic_add(pos_out_ptr, (pos_row_bc, cols), sum5_v)


@oracle_impl(hardware="B200", point="0608152d")
def oracle_forward(inputs):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
        *_shape_params,
    ) = inputs
    device = arg2.device
    rows = int(arg2.numel() // arg2.shape[-1])
    hidden = int(arg2.shape[-1])
    assert hidden == HIDDEN and rows == ROWS

    # Prep row-flat views (no copy).
    arg1_flat = arg1.view(ROWS, HIDDEN).contiguous()
    arg2_flat = arg2.view(ROWS, HIDDEN).contiguous()
    arg3_flat = arg3.view(ROWS, HIDDEN).contiguous()
    arg4_flat = arg4.view(ROWS, HIDDEN).contiguous()
    arg5_flat = arg5.view(ROWS, HIDDEN).contiguous()
    arg6_flat = arg6.view(HIDDEN).contiguous()  # weight
    arg7_flat = arg7.view(ROWS, HIDDEN).contiguous()  # normed
    arg8_flat = arg8.view(ROWS).contiguous()  # scale

    grad_out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.float32)
    sum_5_2d = torch.zeros((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_x_normed = torch.empty((NUM_GROUPS, HIDDEN_PAD), device=device, dtype=torch.float32)
    partial_x = torch.empty((NUM_GROUPS, HIDDEN_PAD), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (NUM_GROUPS, 1, 1),
        _row_ln_backward_kernel,
        (arg1_flat, arg2_flat, arg3_flat, arg4_flat, arg5_flat,
         arg6_flat, arg7_flat, arg8_flat,
         grad_out, sum_5_2d,
         partial_x_normed, partial_x,
         ROWS_PER_GROUP, HIDDEN, HIDDEN_PAD, SEQ, ROWS, DROP_SCALE),
    )

    # Finalize column reductions.
    sum_3_pad = torch.empty((HIDDEN_PAD,), device=device, dtype=torch.float32)
    sum_4_pad = torch.empty((HIDDEN_PAD,), device=device, dtype=torch.float32)
    num_groups_pad = _next_pow2(NUM_GROUPS)
    ct.launch(
        stream, (1, 1, 1),
        _finalize_partials_kernel,
        (partial_x_normed, partial_x, sum_3_pad, sum_4_pad,
         NUM_GROUPS, num_groups_pad, HIDDEN_PAD),
    )
    sum_3 = sum_3_pad[:HIDDEN].contiguous()
    sum_4 = sum_4_pad[:HIDDEN].contiguous()

    # Destinations
    pos_out = torch.zeros((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    segment_out = torch.zeros((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = arg0.to(torch.float32).contiguous().clone()  # base + accumulate

    # Indices
    pos_idx = arg9.view(SEQ)
    seg_idx = arg10.view(SEQ).unsqueeze(0).expand(BATCH, SEQ).reshape(ROWS).contiguous()
    voc_idx = arg11.view(ROWS)

    h_tiles = HIDDEN // BLOCK_H
    ct.launch(
        stream, (ROWS, h_tiles, 1), _scatter_seg_voc_kernel,
        (grad_out, seg_idx, voc_idx, segment_out, vocab_out, HIDDEN, BLOCK_H),
    )
    ct.launch(
        stream, (SEQ, h_tiles, 1), _scatter_pos_kernel,
        (sum_5_2d, pos_idx, pos_out, HIDDEN, BLOCK_H),
    )

    return sum_3, sum_4, pos_out, segment_out, vocab_out
