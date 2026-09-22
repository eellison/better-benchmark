"""cuTile port of sum_sum_sum_ed482d857957: GoogleFnet LN backward + embedding grad scatter.

Structure follows the Triton reference's 3-kernel plan:
  Kernel 1 (_init_kernel): zero-init pos_out, seg_out; copy arg7 -> vocab_out.
  Kernel 2 (_row_kernel): per group of ROWS_PER_GROUP rows, compute weighted,
    row_sum, row_dot, grad; accumulate acc_x_normed and acc_x per-group column
    partials; scatter-add grad into pos_out, segment_out, vocab_out.
  Kernel 3 (_finalize_kernel): reduce (num_groups, HIDDEN) partials into
    (HIDDEN,) out_x_normed and out_x.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 768
BLOCK_H = 1024  # next_pow2(768)
POSITION_ROWS = 512
SEGMENT_ROWS = 4
VOCAB_ROWS = 32000


@ct.kernel
def _init_kernel(
    base_vocab_ptr,   # f32 (VOCAB_ROWS * HIDDEN,)
    pos_out_ptr,      # f32 (POS_ROWS * HIDDEN,)
    seg_out_ptr,      # f32 (SEG_ROWS * HIDDEN,)
    vocab_out_ptr,    # f32 (VOCAB_ROWS * HIDDEN,)
    VOCAB_N: ct.Constant[int],
    POS_N: ct.Constant[int],
    SEG_N: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    vocab_mask = offs < VOCAB_N
    values = ct.gather(base_vocab_ptr, offs, mask=vocab_mask, padding_value=0.0)
    ct.scatter(vocab_out_ptr, offs, values, mask=vocab_mask)
    pos_mask = offs < POS_N
    zero_pos = ct.zeros((BLOCK,), dtype=ct.float32)
    ct.scatter(pos_out_ptr, offs, zero_pos, mask=pos_mask)
    seg_mask = offs < SEG_N
    ct.scatter(seg_out_ptr, offs, zero_pos, mask=seg_mask)


@ct.kernel
def _row_kernel(
    x_ptr,             # f32 (ROWS, HIDDEN)
    weight_ptr,        # f32 (HIDDEN,)
    normed_ptr,        # f32 (ROWS, HIDDEN)
    row_scale_ptr,     # f32 (ROWS,)
    pos_index_ptr,     # i64 (SEQ,)
    segment_index_ptr, # i64 (ROWS,)
    vocab_index_ptr,   # i64 (ROWS,)
    partial_x_normed,  # f32 (NUM_GROUPS, HIDDEN)  (per-group column accumulator)
    partial_x,         # f32 (NUM_GROUPS, HIDDEN)
    pos_out_ptr,       # f32 (POSITION_ROWS * HIDDEN,)
    segment_out_ptr,   # f32 (SEGMENT_ROWS * HIDDEN,)
    vocab_out_ptr,     # f32 (VOCAB_ROWS * HIDDEN,)
    HIDDEN_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
    ROW_FACTOR: ct.Constant[float],
    ROWS_PER_GROUP: ct.Constant[int],
):
    group = ct.bid(0)
    cols = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = cols < HIDDEN_
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H_))
    zero_1d = ct.zeros((BLOCK_H_,), dtype=ct.float32)
    zero_2d = ct.zeros((1, BLOCK_H_), dtype=ct.float32)
    zero_scalar = ct.zeros((1,), dtype=ct.int64)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.where(col_mask, weight, zero_1d)
    weight_2d = ct.reshape(weight, (1, BLOCK_H_))

    # Accumulators (start at zero).
    acc_xn = ct.zeros((BLOCK_H_,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_H_,), dtype=ct.float32)

    for row_offset in ct.static_iter(range(ROWS_PER_GROUP)):
        row = group * ROWS_PER_GROUP + row_offset

        x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                    padding_mode=ct.PaddingMode.ZERO)
        normed = ct.load(normed_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                         padding_mode=ct.PaddingMode.ZERO)
        row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
        row_scale_2d = ct.reshape(row_scale, (1, 1))

        weighted = x * weight_2d
        scaled_weighted = weighted * ROW_FACTOR
        weighted_masked = ct.where(col_mask_2d, weighted, zero_2d)
        row_sum = ct.sum(weighted_masked, axis=1, keepdims=True)
        weighted_normed = weighted * normed
        weighted_normed_masked = ct.where(col_mask_2d, weighted_normed, zero_2d)
        row_dot = ct.sum(weighted_normed_masked, axis=1, keepdims=True)
        centered = scaled_weighted - row_sum - normed * row_dot
        grad = row_scale_2d * centered
        grad_flat = ct.reshape(grad, (BLOCK_H_,))

        # Column partials (as 1D vectors).
        acc_xn = acc_xn + ct.where(col_mask, ct.reshape(x * normed, (BLOCK_H_,)), zero_1d)
        acc_x = acc_x + ct.where(col_mask, ct.reshape(x, (BLOCK_H_,)), zero_1d)

        # Scatter-add grad to pos_out at pos_index[token].
        token = row - (row // SEQ_) * SEQ_
        pos_row = ct.gather(pos_index_ptr, ct.full((1,), token, dtype=ct.int32),
                             padding_value=-1)
        pos_active_scalar = (pos_row >= 0) & (pos_row < POSITION_ROWS) & (pos_row != -1)
        pos_active_2d = ct.reshape(pos_active_scalar, (1, 1)) & col_mask_2d
        safe_pos_row = ct.where(pos_active_scalar, pos_row, zero_scalar)
        pos_off = ct.reshape(safe_pos_row, (1, 1)) * HIDDEN_ + ct.reshape(cols, (1, BLOCK_H_))
        grad_pos = ct.where(pos_active_2d, grad, zero_2d)
        ct.atomic_add(pos_out_ptr, (ct.reshape(pos_off, (BLOCK_H_,)),),
                      ct.reshape(grad_pos, (BLOCK_H_,)))

        segment_row = ct.gather(segment_index_ptr, ct.full((1,), row, dtype=ct.int32),
                                 padding_value=-1)
        seg_active_scalar = (segment_row >= 0) & (segment_row < SEGMENT_ROWS) & (segment_row != -1)
        seg_active_2d = ct.reshape(seg_active_scalar, (1, 1)) & col_mask_2d
        safe_seg_row = ct.where(seg_active_scalar, segment_row, zero_scalar)
        seg_off = ct.reshape(safe_seg_row, (1, 1)) * HIDDEN_ + ct.reshape(cols, (1, BLOCK_H_))
        grad_seg = ct.where(seg_active_2d, grad, zero_2d)
        ct.atomic_add(segment_out_ptr, (ct.reshape(seg_off, (BLOCK_H_,)),),
                      ct.reshape(grad_seg, (BLOCK_H_,)))

        vocab_row = ct.gather(vocab_index_ptr, ct.full((1,), row, dtype=ct.int32),
                              padding_value=3)
        vocab_active_scalar = (vocab_row >= 0) & (vocab_row < VOCAB_ROWS) & (vocab_row != 3)
        vocab_active_2d = ct.reshape(vocab_active_scalar, (1, 1)) & col_mask_2d
        safe_vocab_row = ct.where(vocab_active_scalar, vocab_row, zero_scalar)
        vocab_off = ct.reshape(safe_vocab_row, (1, 1)) * HIDDEN_ + ct.reshape(cols, (1, BLOCK_H_))
        grad_vocab = ct.where(vocab_active_2d, grad, zero_2d)
        ct.atomic_add(vocab_out_ptr, (ct.reshape(vocab_off, (BLOCK_H_,)),),
                      ct.reshape(grad_vocab, (BLOCK_H_,)))

    # Store per-group partials (as (1, BLOCK_H_) tiles).
    ct.store(partial_x_normed, index=(group, 0), tile=ct.reshape(acc_xn, (1, BLOCK_H_)))
    ct.store(partial_x, index=(group, 0), tile=ct.reshape(acc_x, (1, BLOCK_H_)))


@ct.kernel
def _finalize_kernel(
    partial_x_normed,   # f32 (NUM_GROUPS, BLOCK_H_)
    partial_x,          # f32 (NUM_GROUPS, BLOCK_H_)
    out_x_normed_ptr,   # f32 (HIDDEN,)
    out_x_ptr,          # f32 (HIDDEN,)
    NUM_GROUPS: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
    GROUP_BLOCK: ct.Constant[int],  # pow2 >= NUM_GROUPS
    FINAL_BLOCK_H: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Load a (GROUP_BLOCK, FINAL_BLOCK_H) sub-tile out of (NUM_GROUPS, BLOCK_H_).
    p_xn = ct.load(partial_x_normed, index=(0, col_block),
                   shape=(GROUP_BLOCK, FINAL_BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    p_x = ct.load(partial_x, index=(0, col_block),
                  shape=(GROUP_BLOCK, FINAL_BLOCK_H),
                  padding_mode=ct.PaddingMode.ZERO)
    # Only valid group rows (< NUM_GROUPS) contribute; the load pads with zeros for the tail.
    s_xn = ct.sum(p_xn, axis=0)
    s_x = ct.sum(p_x, axis=0)

    # Write to output. Output space is HIDDEN=768; last tile has some padded slots which
    # we mask off at store time by using the leading valid FINAL_BLOCK_H columns.
    # If col_block * FINAL_BLOCK_H + FINAL_BLOCK_H > HIDDEN, the store extends OOB;
    # caller enforces HIDDEN % FINAL_BLOCK_H == 0 (768 % 8 == 0).
    ct.store(out_x_normed_ptr, index=(col_block,), tile=s_xn)
    ct.store(out_x_ptr, index=(col_block,), tile=s_x)


def _ceil_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="f69e9b69", ROWS_PER_GROUP=8, FINAL_BLOCK_H=8, INIT_BLOCK=1024)
def oracle_forward(inputs, *, ROWS_PER_GROUP, FINAL_BLOCK_H, INIT_BLOCK):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
        *_shape_params,
    ) = inputs
    device = arg0.device
    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])

    # Init destinations.
    pos_out = torch.empty((POSITION_ROWS, hidden), device=device, dtype=torch.float32)
    segment_out = torch.empty((SEGMENT_ROWS, hidden), device=device, dtype=torch.float32)
    vocab_out = torch.empty_strided(
        tuple(int(dim) for dim in arg7.shape),
        tuple(int(stride) for stride in arg7.stride()),
        device=device, dtype=torch.float32,
    )
    num_groups = (rows + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP
    partial_x_normed = torch.empty((num_groups, hidden), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_groups, hidden), device=device, dtype=torch.float32)
    out_x_normed = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_x = torch.empty((hidden,), device=device, dtype=torch.float32)

    vocab_n = VOCAB_ROWS * hidden
    pos_n = POSITION_ROWS * hidden
    seg_n = SEGMENT_ROWS * hidden

    x_2d = arg0.view(rows, hidden)
    normed_2d = arg2.reshape(rows, hidden)
    row_scale_1d = arg3.reshape(rows)
    pos_idx_1d = arg4.reshape(SEQ)
    seg_idx_expanded = arg5.expand(BATCH, SEQ).reshape(rows).contiguous()
    vocab_idx_1d = arg6.reshape(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, ((vocab_n + INIT_BLOCK - 1) // INIT_BLOCK, 1, 1), _init_kernel,
        (
            arg7.reshape(-1),
            pos_out.reshape(-1), segment_out.reshape(-1), vocab_out.reshape(-1),
            vocab_n, pos_n, seg_n, INIT_BLOCK,
        ),
    )
    ct.launch(
        stream, (num_groups, 1, 1), _row_kernel,
        (
            x_2d, arg1, normed_2d, row_scale_1d,
            pos_idx_1d, seg_idx_expanded, vocab_idx_1d,
            partial_x_normed, partial_x,
            pos_out.reshape(-1), segment_out.reshape(-1), vocab_out.reshape(-1),
            hidden, SEQ, BLOCK_H, float(hidden), ROWS_PER_GROUP,
        ),
    )
    group_block = _ceil_pow2(num_groups)
    if hidden % FINAL_BLOCK_H != 0:
        raise NotImplementedError("FINAL_BLOCK_H must divide HIDDEN")
    ct.launch(
        stream, (hidden // FINAL_BLOCK_H, 1, 1), _finalize_kernel,
        (
            partial_x_normed, partial_x,
            out_x_normed, out_x,
            num_groups, hidden, BLOCK_H, group_block, FINAL_BLOCK_H,
        ),
    )

    return out_x_normed, out_x, pos_out, segment_out, vocab_out
