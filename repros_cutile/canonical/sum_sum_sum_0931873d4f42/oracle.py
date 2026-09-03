"""cuTile port of sum_sum_sum_0931873d4f42: DeiT class-token select-scatter LN backward.

Only token 0 (of 197) is active — all other rows are zero. We compute LN
backward for the active rows and channel sums, then zero-fill inactive rows.
HIDDEN=192 non-power-of-2; use BLOCK_C=256 with column-masked reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
HIDDEN = 192
ROWS = BATCH * TOKENS


@ct.kernel
def _token0_rows_kernel(
    source_ptr,     # bf16 [BATCH, HIDDEN]
    gamma_ptr,      # f32 [HIDDEN]
    xhat_ptr,       # f32 [BATCH*TOKENS, HIDDEN]
    scale_ptr,      # f32 [BATCH*TOKENS]
    out_f32_row_ptr, # f32 [BATCH, BLOCK_C]  (only token 0 written; other rows caller-zeroed)
    out_bf16_row_ptr, # bf16 [BATCH, BLOCK_C]
    partial_sum3_ptr, # f32 [BATCH, BLOCK_C]
    partial_sum4_ptr, # f32 [BATCH, BLOCK_C]
    partial_sum5_ptr, # f32 [BATCH, BLOCK_C]
    HIDDEN_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    batch = ct.bid(0)

    source_bf = ct.load(source_ptr, index=(batch, 0), shape=(1, BLOCK_C),
                        padding_mode=ct.PaddingMode.ZERO)
    source = ct.astype(source_bf, ct.float32)
    gamma = ct.load(gamma_ptr, index=(0,), shape=(BLOCK_C,),
                    padding_mode=ct.PaddingMode.ZERO)
    xhat = ct.load(xhat_ptr, index=(batch * TOKENS, 0), shape=(1, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    scale = ct.load(scale_ptr, index=(batch * TOKENS,), shape=(1,))

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = cols < HIDDEN_C
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_C))
    zero = ct.full((1, BLOCK_C), 0.0, dtype=ct.float32)

    gamma_2d = ct.reshape(gamma, (1, BLOCK_C))
    weighted = source * gamma_2d
    weighted_m = ct.where(col_mask_2d, weighted, zero)
    row_sum = ct.sum(weighted_m)

    weighted_xhat = weighted * xhat
    weighted_xhat_m = ct.where(col_mask_2d, weighted_xhat, zero)
    row_dot = ct.sum(weighted_xhat_m)

    scale_2d = ct.reshape(scale, (1, 1))
    centered = weighted * float(HIDDEN_C) - row_sum - xhat * row_dot
    grad = scale_2d * centered
    grad_bf = ct.astype(grad, ct.bfloat16)

    ct.store(out_f32_row_ptr, index=(batch, 0), tile=grad)
    ct.store(out_bf16_row_ptr, index=(batch, 0), tile=grad_bf)

    p3 = ct.where(col_mask_2d, source * xhat, zero)
    p4 = ct.where(col_mask_2d, source, zero)
    grad_bf_f32 = ct.astype(grad_bf, ct.float32)
    p5 = ct.where(col_mask_2d, grad_bf_f32, zero)
    ct.store(partial_sum3_ptr, index=(batch, 0), tile=p3)
    ct.store(partial_sum4_ptr, index=(batch, 0), tile=p4)
    ct.store(partial_sum5_ptr, index=(batch, 0), tile=p5)


@ct.kernel
def _finalize_channel_sums_kernel(
    partial_sum3_ptr,   # f32 [BATCH, BLOCK_C]
    partial_sum4_ptr,   # f32 [BATCH, BLOCK_C]
    partial_sum5_ptr,   # f32 [BATCH, BLOCK_C]
    out_sum3_ptr,       # f32 [HIDDEN]
    out_sum4_ptr,       # f32 [HIDDEN]
    out_sum5_ptr,       # f32 [HIDDEN]
    ROWS_: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    vals3 = ct.load(partial_sum3_ptr, index=(0, col_block), shape=(ROWS_, FINAL_BLOCK_C))
    vals4 = ct.load(partial_sum4_ptr, index=(0, col_block), shape=(ROWS_, FINAL_BLOCK_C))
    vals5 = ct.load(partial_sum5_ptr, index=(0, col_block), shape=(ROWS_, FINAL_BLOCK_C))
    sum3 = ct.sum(vals3, axis=0)
    sum4 = ct.sum(vals4, axis=0)
    sum5_f = ct.sum(vals5, axis=0)
    sum5 = ct.astype(ct.astype(sum5_f, ct.bfloat16), ct.float32)
    ct.store(out_sum3_ptr, index=(col_block,), tile=sum3)
    ct.store(out_sum4_ptr, index=(col_block,), tile=sum4)
    ct.store(out_sum5_ptr, index=(col_block,), tile=sum5)


@oracle_impl(
    hardware="B200",
    point="7639f983",
    BLOCK_C=256,
    ZERO_BLOCK=1024,
    FINAL_BLOCK_C=16,
)
def oracle_forward(inputs, *, BLOCK_C: int, ZERO_BLOCK: int, FINAL_BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    device = arg0_1.device

    # Allocate zero-initialized output tensors — inactive rows stay zero.
    out_f32 = torch.zeros(
        (BATCH, TOKENS, HIDDEN),
        device=device,
        dtype=torch.float32,
    )
    out_bf16_3d = torch.zeros(
        (BATCH, TOKENS, HIDDEN),
        device=device,
        dtype=torch.bfloat16,
    )

    padded_row_f32 = torch.empty((BATCH, BLOCK_C), device=device, dtype=torch.float32)
    padded_row_bf16 = torch.empty((BATCH, BLOCK_C), device=device, dtype=torch.bfloat16)
    p3 = torch.empty((BATCH, BLOCK_C), device=device, dtype=torch.float32)
    p4 = torch.empty((BATCH, BLOCK_C), device=device, dtype=torch.float32)
    p5 = torch.empty((BATCH, BLOCK_C), device=device, dtype=torch.float32)

    # xhat/scale are 3D [BATCH, TOKENS, ...]; view as flat.
    xhat_2d = arg2_1.view(ROWS, HIDDEN)
    scale_1d = arg3_1.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, 1, 1),
        _token0_rows_kernel,
        (arg0_1, arg1_1, xhat_2d, scale_1d,
         padded_row_f32, padded_row_bf16,
         p3, p4, p5,
         HIDDEN, BLOCK_C),
    )

    # Fill token-0 slice with computed values (all other tokens are zero).
    out_f32[:, 0, :] = padded_row_f32[:, :HIDDEN]
    out_bf16_3d[:, 0, :] = padded_row_bf16[:, :HIDDEN]

    out_bf16 = out_bf16_3d.view(ROWS, HIDDEN)

    # Column reductions inside a cuTile kernel (mirrors Triton's
    # _finalize_channel_sums_kernel). BLOCK_C partial cols beyond HIDDEN
    # were masked to zero by the main kernel, so summing across them is safe.
    sum3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum5 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        ((HIDDEN + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _finalize_channel_sums_kernel,
        (p3, p4, p5, sum3, sum4, sum5, BATCH, FINAL_BLOCK_C),
    )

    return out_f32, sum3, sum4, out_bf16, out_bf16.permute(1, 0), sum5
