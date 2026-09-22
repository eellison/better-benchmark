"""cuTile port of sum_sum_sum_f6eef8c5a35a: GPT2 LN-backward + embedding scatter.

Uses cuTile for the fused LN-backward per-row kernel; torch handles the
_unsafe_masked_index_put_accumulate scatter.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
BATCH_ROWS = 8 * 1024  # 8192


@ct.kernel
def _ln_backward_row_kernel(
    grad_ptr,       # f32 [rows, HIDDEN]  view (from arg0 f32-cast)
    weight_ptr,     # f32 [HIDDEN]         arg1
    input_ptr,      # f32 [rows, HIDDEN]  add = arg2 + arg3
    mask_ptr,       # b8 [rows, HIDDEN]   arg4
    mean_ptr,       # f32 [rows, 1]       arg5
    invstd_ptr,     # f32 [rows, 1]       arg6
    add_bg_ptr,     # f32 [rows, HIDDEN]  arg7
    add_out_ptr,    # f32 [rows, HIDDEN]  add_1 = arg7 + mul_7
    sum3_partial_ptr, # f32 [rows_chunks, HIDDEN]  partial per-chunk of sum_3
    sum4_partial_ptr, # f32 [rows_chunks, HIDDEN]  partial per-chunk of sum_4
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    r_block = ct.bid(0)

    grad = ct.load(grad_ptr, index=(r_block, 0), shape=(BLOCK_R, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    input_v = ct.load(input_ptr, index=(r_block, 0), shape=(BLOCK_R, BLOCK_H),
                      padding_mode=ct.PaddingMode.ZERO)
    mask = ct.load(mask_ptr, index=(r_block, 0), shape=(BLOCK_R, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(r_block, 0), shape=(BLOCK_R, 1),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(r_block, 0), shape=(BLOCK_R, 1),
                     padding_mode=ct.PaddingMode.ZERO)
    add_bg = ct.load(add_bg_ptr, index=(r_block, 0), shape=(BLOCK_R, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)

    ri = ct.arange(BLOCK_R, dtype=ct.int32) + r_block * BLOCK_R
    ci = ct.arange(BLOCK_H, dtype=ct.int32)
    row_ok_1d = ri < ROWS_
    col_ok_1d = ci < HIDDEN_
    row_ok = ct.reshape(row_ok_1d, (BLOCK_R, 1))
    col_ok = ct.reshape(col_ok_1d, (1, BLOCK_H))
    v_mask = row_ok & col_ok
    zero_f = ct.full((BLOCK_R, BLOCK_H), 0.0, dtype=ct.float32)

    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    mul = grad * weight_2d
    mul = ct.where(v_mask, mul, zero_f)
    mul_1 = mul * HIDDEN_
    sum_1_ = ct.sum(mul, axis=1, keepdims=True)  # (BLOCK_R, 1)

    mask_f = ct.astype(mask, ct.float32)
    add_masked = mask_f * input_v
    mul_3 = add_masked * 1.1111111111111112
    sub_ = mul_3 - mean
    mul_4 = sub_ * invstd
    mul_5 = mul * mul_4
    sum_2_ = ct.sum(mul_5, axis=1, keepdims=True)
    mul_6 = mul_4 * sum_2_
    sub_1 = mul_1 - sum_1_
    sub_2 = sub_1 - mul_6
    div_ = invstd * (1.0 / HIDDEN_)
    mul_7 = div_ * sub_2

    add_out = add_bg + mul_7
    ct.store(add_out_ptr, index=(r_block, 0), tile=add_out)

    # Emit partial per-chunk contributions: sum_3 = sum(mul_8 = grad * mul_4, dim=[0,1])
    # → per-column partial from this row chunk.
    mul_8 = grad * mul_4
    mul_8 = ct.where(v_mask, mul_8, zero_f)
    p_sum3 = ct.sum(mul_8, axis=0)
    ct.store(sum3_partial_ptr, index=(r_block, 0),
             tile=ct.reshape(p_sum3, (1, BLOCK_H)))
    # sum_4 = sum(grad, dim=[0,1]) per column
    grad_m = ct.where(v_mask, grad, zero_f)
    p_sum4 = ct.sum(grad_m, axis=0)
    ct.store(sum4_partial_ptr, index=(r_block, 0),
             tile=ct.reshape(p_sum4, (1, BLOCK_H)))


@ct.kernel
def _reduce_chunks_kernel(
    partial_ptr,
    out_ptr,
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    p = ct.load(partial_ptr, index=(0, c_block), shape=(BLOCK_CHUNKS, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    ri = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    ci = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    row_ok_1d = ri < NUM_CHUNKS
    col_ok_1d = ci < C_
    row_ok = ct.reshape(row_ok_1d, (BLOCK_CHUNKS, 1))
    col_ok = ct.reshape(col_ok_1d, (1, BLOCK_C))
    m = row_ok & col_ok
    zero_f = ct.full((BLOCK_CHUNKS, BLOCK_C), 0.0, dtype=ct.float32)
    p = ct.where(m, p, zero_f)
    s = ct.sum(p, axis=0)
    ct.store(out_ptr, index=(c_block,), tile=s)


@ct.kernel
def _scatter_kernel(
    add1_ptr,       # f32 [BATCH_ROWS, HIDDEN_PAD]  add_out_pad
    keep_ptr,       # bool [BATCH_ROWS, HIDDEN_PAD]  arg4_pad
    pos_idx_ptr,    # i64 [SEQ]                     arg8
    vocab_idx_ptr,  # i64 [BATCH_ROWS]              arg9
    pos_out_ptr,    # f32 [POSITION_ROWS, HIDDEN]
    vocab_out_ptr,  # f32 [VOCAB_ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    HIDDEN_PAD_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    VOCAB_ROWS_: ct.Constant[int],
    DROP_SCALE_: ct.Constant[float],
):
    row = ct.bid(0)  # 0..BATCH_ROWS
    add1 = ct.load(add1_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
                   padding_mode=ct.PaddingMode.ZERO)
    keep = ct.astype(
        ct.load(keep_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mul_10 = add1 * keep * DROP_SCALE_  # f32 [1, HIDDEN_PAD]

    cols_1d = ct.arange(HIDDEN_PAD_, dtype=ct.int32)
    col_valid_2d = ct.reshape(cols_1d < HIDDEN_, (1, HIDDEN_PAD_))
    cols_2d = ct.reshape(cols_1d, (1, HIDDEN_PAD_))
    neg_ones = ct.full((1, HIDDEN_PAD_), -1, dtype=ct.int32)
    cols_masked = ct.where(col_valid_2d, cols_2d, neg_ones)

    # For pos scatter: arg8_1 is shape [1, SEQ] with indices in [0, POSITION_ROWS).
    # Each token t = row % SEQ maps to pos_idx[t]. Original torch code sums mul_10
    # over dim=0 (batch) then scatters, so each unique token gets summed contributions
    # from all batches. Atomic_add achieves the same.
    token = row - (row // SEQ_) * SEQ_
    pos_row_scalar = ct.load(pos_idx_ptr, index=(token,), shape=(1,),
                              padding_mode=ct.PaddingMode.ZERO)
    pos_row_bc = ct.broadcast_to(
        ct.astype(ct.reshape(pos_row_scalar, (1,)), ct.int32), (1, HIDDEN_PAD_)
    )
    # Also mask by check_bounds; but ensure the row is not -1.
    ct.atomic_add(pos_out_ptr, (pos_row_bc, cols_masked), mul_10)

    # For vocab scatter: arg9_1 has -1 sentinels; only accumulate valid rows.
    vocab_row_scalar = ct.load(vocab_idx_ptr, index=(row,), shape=(1,),
                                padding_mode=ct.PaddingMode.ZERO)
    vocab_row = ct.astype(ct.reshape(vocab_row_scalar, (1,)), ct.int32)
    # Filter -1 rows by remapping to negative.
    vocab_row_bc = ct.broadcast_to(vocab_row, (1, HIDDEN_PAD_))
    ct.atomic_add(vocab_out_ptr, (vocab_row_bc, cols_masked), mul_10)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="7e991e0f")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, *_shape) = inputs
    device = arg0_1.device

    # Precompute inputs shared by the kernel.
    grad = arg0_1.to(torch.float32).view(8, 1024, HIDDEN).view(BATCH_ROWS, HIDDEN)
    input_v = (arg2_1 + arg3_1).view(BATCH_ROWS, HIDDEN)
    mask = arg4_1.view(BATCH_ROWS, HIDDEN)
    mean = arg5_1.view(BATCH_ROWS, 1)
    invstd = arg6_1.view(BATCH_ROWS, 1)
    add_bg = arg7_1.view(BATCH_ROWS, HIDDEN)

    BLOCK_R = 8
    BLOCK_H = 1024  # covers HIDDEN=768
    num_chunks = (BATCH_ROWS + BLOCK_R - 1) // BLOCK_R
    block_chunks = _next_pow2(num_chunks)

    # Pad the row dim in outputs & intermediates
    add_out = torch.empty((BATCH_ROWS, HIDDEN), device=device, dtype=torch.float32)
    sum3_partial = torch.empty((num_chunks, BLOCK_H), device=device, dtype=torch.float32)
    sum4_partial = torch.empty((num_chunks, BLOCK_H), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    # Pad add_out storage to allow BLOCK_H stores past HIDDEN
    add_out_pad = torch.empty((BATCH_ROWS, BLOCK_H), device=device, dtype=torch.float32)
    # Also pad grad and other inputs to BLOCK_H
    def _pad(t, dtype):
        p = torch.zeros((BATCH_ROWS, BLOCK_H), device=device, dtype=dtype)
        p[:, :HIDDEN].copy_(t)
        return p
    grad_pad = _pad(grad, torch.float32)
    input_pad = _pad(input_v, torch.float32)
    mask_pad = _pad(mask, torch.bool)
    weight_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    weight_pad[:HIDDEN].copy_(arg1_1)
    add_bg_pad = _pad(add_bg, torch.float32)

    ct.launch(
        stream,
        (num_chunks, 1, 1),
        _ln_backward_row_kernel,
        (grad_pad, weight_pad, input_pad, mask_pad, mean, invstd, add_bg_pad,
         add_out_pad,
         sum3_partial, sum4_partial,
         BATCH_ROWS, HIDDEN, BLOCK_R, BLOCK_H),
    )
    add_out_final = add_out_pad[:, :HIDDEN].contiguous().view(8, 1024, HIDDEN)

    # Reduce the per-chunk partials to get sum_3, sum_4 (both [HIDDEN]).
    sum_3 = torch.empty((BLOCK_H,), device=device, dtype=torch.float32)
    sum_4 = torch.empty((BLOCK_H,), device=device, dtype=torch.float32)
    ct.launch(stream, (1, 1, 1), _reduce_chunks_kernel,
              (sum3_partial, sum_3, num_chunks, block_chunks, HIDDEN, BLOCK_H))
    ct.launch(stream, (1, 1, 1), _reduce_chunks_kernel,
              (sum4_partial, sum_4, num_chunks, block_chunks, HIDDEN, BLOCK_H))
    sum_3 = sum_3[:HIDDEN]
    sum_4 = sum_4[:HIDDEN]

    # Scatter kernel: computes mul_10 = add_out * arg4 * DROP_SCALE inside the
    # kernel and atomically accumulates into pos_out and vocab_out. This mirrors
    # Triton's `_row_scatter_partials_kernel` scatter emissions.
    imp_1 = torch.zeros((1024, HIDDEN), dtype=torch.float32, device=device)
    imp_2 = torch.zeros((50257, HIDDEN), dtype=torch.float32, device=device)
    pos_idx = arg8_1.view(1024).contiguous()  # [SEQ]
    vocab_idx = arg9_1.view(BATCH_ROWS).contiguous()  # [ROWS]

    ct.launch(
        stream,
        (BATCH_ROWS, 1, 1),
        _scatter_kernel,
        (add_out_pad, mask_pad, pos_idx, vocab_idx, imp_1, imp_2,
         HIDDEN, BLOCK_H, 1024, 50257, 1.1111111111111112),
    )

    return sum_3, sum_4, imp_1, imp_2
