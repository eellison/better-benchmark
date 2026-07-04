"""cuTile port of sum_sum_sum_d90a5c698770: DeiT LN-backward + select-scatter.

Two-kernel structure (Triton has 3; we merge the token-rows and
zero-inactive-rows into one dense loop and finalize channel sums separately):
  - _row_kernel: for every (batch, token) row: compute add + LN-backward, store,
    and emit per-row partials for the 3 column reductions.
  - _finalize_channel_sums_kernel: reduce partial column sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 198
HIDDEN = 768
ROWS = BATCH * TOKENS
ROW_FACTOR = 768.0


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _row_kernel(
    arg0_ptr,          # bf16 [BATCH, HIDDEN] — token-1 source
    arg1_ptr,          # bf16 [BATCH, HIDDEN] — token-0 source
    weight_ptr,        # f32 [HIDDEN]
    xhat_ptr,          # f32 [ROWS, HIDDEN] (viewed from arg3_1)
    invstd_ptr,        # f32 [ROWS] (viewed from arg4_1)
    out_f32_ptr,       # f32 [ROWS, HIDDEN]
    out_bf16_ptr,      # bf16 [ROWS, HIDDEN]
    partial_dot_ptr,   # f32 [ROWS, HIDDEN] (per-row: add * xhat)
    partial_add_ptr,   # f32 [ROWS, HIDDEN] (per-row: add)
    partial_bf16_ptr,  # bf16 [ROWS, HIDDEN] (per-row: grad_bf16)
    HIDDEN_C: ct.Constant[int],
    H_PAD: ct.Constant[int],
    TOKENS_C: ct.Constant[int],
):
    # grid = (ROWS,)
    row = ct.bid(0)
    batch = row // TOKENS_C
    token = row - batch * TOKENS_C

    weight = ct.load(weight_ptr, index=(0,), shape=(H_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    xhat = ct.load(xhat_ptr, index=(row, 0), shape=(1, H_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))

    cols = ct.arange(H_PAD, dtype=ct.int32)
    col_mask = cols < HIDDEN_C
    col_mask_2d = ct.reshape(col_mask, (1, H_PAD))
    zero_2d = ct.full((1, H_PAD), 0.0, dtype=ct.float32)

    # Load add source: 0 for most tokens; arg1[batch] for token 0; arg0[batch] for token 1.
    a0 = ct.load(arg0_ptr, index=(batch, 0), shape=(1, H_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    a1 = ct.load(arg1_ptr, index=(batch, 0), shape=(1, H_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    a0_f = ct.astype(a0, ct.float32)
    a1_f = ct.astype(a1, ct.float32)
    is_token_0 = token == 0
    is_token_1 = token == 1
    add = ct.where(is_token_0, a1_f, ct.where(is_token_1, a0_f, zero_2d))
    add_masked = ct.where(col_mask_2d, add, zero_2d)

    # LN-backward
    weight_2d = ct.reshape(weight, (1, H_PAD))
    mul = add_masked * weight_2d
    mul_masked = ct.where(col_mask_2d, mul, zero_2d)
    mul_hidden = mul * ROW_FACTOR
    row_sum = ct.sum(mul_masked, axis=1, keepdims=True)
    mul_xhat = mul * xhat
    mul_xhat_masked = ct.where(col_mask_2d, mul_xhat, zero_2d)
    row_dot = ct.sum(mul_xhat_masked, axis=1, keepdims=True)
    centered = mul_hidden - row_sum - xhat * row_dot
    invstd_2d = ct.reshape(invstd, (1, 1))
    grad = invstd_2d * centered
    grad_masked = ct.where(col_mask_2d, grad, zero_2d)
    grad_bf16 = ct.astype(grad_masked, ct.bfloat16)

    ct.store(out_f32_ptr, index=(row, 0), tile=grad_masked)
    ct.store(out_bf16_ptr, index=(row, 0), tile=grad_bf16)
    ct.store(partial_dot_ptr, index=(row, 0), tile=add_masked * xhat)
    ct.store(partial_add_ptr, index=(row, 0), tile=add_masked)
    ct.store(partial_bf16_ptr, index=(row, 0), tile=grad_bf16)


@ct.kernel
def _finalize_channel_sums_kernel(
    partial_dot_ptr,   # f32 [ROWS, HIDDEN]
    partial_add_ptr,   # f32 [ROWS, HIDDEN]
    partial_bf16_ptr,  # bf16 [ROWS, HIDDEN]
    out_sum3_ptr,      # f32 [HIDDEN]
    out_sum4_ptr,      # f32 [HIDDEN]
    out_sum5_ptr,      # f32 [HIDDEN]
    ROWS_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
):
    # grid = (cdiv(HIDDEN, FINAL_BLOCK_C),)
    ht = ct.bid(0)
    tile_dot = ct.load(partial_dot_ptr, index=(0, ht),
                       shape=(ROWS_C, FINAL_BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    tile_add = ct.load(partial_add_ptr, index=(0, ht),
                       shape=(ROWS_C, FINAL_BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    tile_bf16 = ct.load(partial_bf16_ptr, index=(0, ht),
                        shape=(ROWS_C, FINAL_BLOCK_C),
                        padding_mode=ct.PaddingMode.ZERO)

    acc_dot = ct.sum(tile_dot, axis=0)
    acc_add = ct.sum(tile_add, axis=0)
    acc_bf16 = ct.sum(ct.astype(tile_bf16, ct.float32), axis=0)
    acc_bf16_r = ct.astype(ct.astype(acc_bf16, ct.bfloat16), ct.float32)

    ct.store(out_sum3_ptr, index=(ht,), tile=acc_dot)
    ct.store(out_sum4_ptr, index=(ht,), tile=acc_add)
    ct.store(out_sum5_ptr, index=(ht,), tile=acc_bf16_r)


@oracle_impl(hardware="B200", point="da5e6ff8", FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, FINAL_BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    device = arg0_1.device

    N, S, H = tuple(int(d) for d in shape0)
    rows = N * S
    H_PAD = _next_pow2(H)

    out_f32 = torch.empty_strided(
        (N, S, H), (S * H, H, 1), device=device, dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (rows, H), (H, 1), device=device, dtype=torch.bfloat16,
    )
    partial_dot = torch.empty((rows, H), device=device, dtype=torch.float32)
    partial_add = torch.empty((rows, H), device=device, dtype=torch.float32)
    partial_bf16 = torch.empty((rows, H), device=device, dtype=torch.bfloat16)

    out_sum3 = torch.empty((H,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((H,), device=device, dtype=torch.float32)
    out_sum5 = torch.empty((H,), device=device, dtype=torch.float32)

    xhat_2d = arg3_1.view(rows, H)
    invstd_1d = arg4_1.view(rows)
    out_f32_2d = out_f32.view(rows, H)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _row_kernel,
        (arg0_1, arg1_1, arg2_1, xhat_2d, invstd_1d,
         out_f32_2d, out_bf16,
         partial_dot, partial_add, partial_bf16,
         H, H_PAD, S),
    )
    ROW_PAD = 1 << (rows - 1).bit_length()  # next power of 2 of rows
    ct.launch(
        stream, (H // FINAL_BLOCK_C, 1, 1),
        _finalize_channel_sums_kernel,
        (partial_dot, partial_add, partial_bf16,
         out_sum3, out_sum4, out_sum5,
         ROW_PAD, H, FINAL_BLOCK_C),
    )

    return out_f32, out_sum3, out_sum4, out_bf16, out_bf16.permute(1, 0), out_sum5
