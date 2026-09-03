"""cuTile port of sum_sum_sum_899373d9c485: LN-backward tail.

Mirrors Triton's non-fused multi-kernel plan (row_stats + store_and_partial +
finalize_channel_sums). For fused shapes we still use 3 kernels — that
LOWERS fusion vs. Triton (2 kernels), which is allowed; we cannot INCREASE
fusion. All reductions live in-kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _pow2_ceil(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@ct.kernel
def _row_stats_kernel(
    x_ptr,          # bf16 [rows, C]
    weight_ptr,     # f32 [C]
    rhs_ptr,        # f32 [rows, C]
    row_sum_ptr,    # f32 [rows]
    row_dot_ptr,    # f32 [rows]
    C_: ct.Constant[int],
    BLOCK_C_PAD: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_C_PAD, dtype=ct.int32)
    c_valid = ct.reshape(cols < C_, (1, BLOCK_C_PAD))

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_C_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_C_PAD),
                  padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_C_PAD))
    x = ct.astype(x_bf, ct.float32)
    weighted = x * weight_2d
    zero_2d = ct.zeros((1, BLOCK_C_PAD), dtype=ct.float32)
    row_sum = ct.sum(ct.where(c_valid, weighted, zero_2d), axis=1)
    row_dot = ct.sum(ct.where(c_valid, weighted * rhs, zero_2d), axis=1)
    ct.store(row_sum_ptr, index=(row,), tile=ct.reshape(row_sum, (1,)))
    ct.store(row_dot_ptr, index=(row,), tile=ct.reshape(row_dot, (1,)))


@ct.kernel
def _store_and_partial_kernel(
    x_ptr,          # bf16 [rows, C]
    weight_ptr,     # f32 [C]
    rhs_ptr,        # f32 [rows, C]
    row_scale_ptr,  # f32 [rows]
    row_sum_ptr,    # f32 [rows]
    row_dot_ptr,    # f32 [rows]
    grad_ptr,       # f32 [rows, C]
    grad_bf16_ptr,  # bf16 [rows, C]
    partials_ptr,   # f32 [num_row_blocks, 3, C]
    C_: ct.Constant[int],
    NORM_FACTOR: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)

    x_bf = ct.load(x_ptr, index=(row, col_block), shape=(1, BLOCK_C))
    rhs = ct.load(rhs_ptr, index=(row, col_block), shape=(1, BLOCK_C))
    weight = ct.load(weight_ptr, index=(col_block,), shape=(BLOCK_C,))
    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    row_sum = ct.load(row_sum_ptr, index=(row,), shape=(1,))
    row_dot = ct.load(row_dot_ptr, index=(row,), shape=(1,))

    x = ct.astype(x_bf, ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    row_scale_2d = ct.reshape(row_scale, (1, 1))
    row_sum_2d = ct.reshape(row_sum, (1, 1))
    row_dot_2d = ct.reshape(row_dot, (1, 1))

    grad = row_scale_2d * (
        x * weight_2d * NORM_FACTOR - row_sum_2d - rhs * row_dot_2d
    )
    grad_bf = ct.astype(grad, ct.bfloat16)

    ct.store(grad_ptr, index=(row, col_block), tile=grad)
    ct.store(grad_bf16_ptr, index=(row, col_block), tile=grad_bf)

    # Per-(row, col_block) column partial contributions.
    # Store into partials[row, i, col_block] where i in {0,1,2}.
    x_rhs_tile = ct.reshape(x * rhs, (1, 1, BLOCK_C))
    x_tile = ct.reshape(x, (1, 1, BLOCK_C))
    grad_bf_f32 = ct.astype(grad_bf, ct.float32)
    grad_tile = ct.reshape(grad_bf_f32, (1, 1, BLOCK_C))

    ct.store(partials_ptr, index=(row, 0, col_block), tile=x_rhs_tile)
    ct.store(partials_ptr, index=(row, 1, col_block), tile=x_tile)
    ct.store(partials_ptr, index=(row, 2, col_block), tile=grad_tile)


@ct.kernel
def _finalize_channel_sums_kernel(
    partials_ptr,   # f32 [rows, 3, C]
    out_x_rhs_ptr,  # f32 [C]
    out_x_ptr,      # f32 [C]
    out_grad_sum_ptr,  # f32 [C]
    ROWS_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)

    acc_x_rhs = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_grad = ct.zeros((BLOCK_C,), dtype=ct.float32)

    for r in range(ROWS_):
        xr = ct.load(partials_ptr, index=(r, 0, col_block), shape=(1, 1, BLOCK_C))
        xv = ct.load(partials_ptr, index=(r, 1, col_block), shape=(1, 1, BLOCK_C))
        gv = ct.load(partials_ptr, index=(r, 2, col_block), shape=(1, 1, BLOCK_C))
        acc_x_rhs = acc_x_rhs + ct.reshape(xr, (BLOCK_C,))
        acc_x = acc_x + ct.reshape(xv, (BLOCK_C,))
        acc_grad = acc_grad + ct.reshape(gv, (BLOCK_C,))

    ct.store(out_x_rhs_ptr, index=(col_block,), tile=acc_x_rhs)
    ct.store(out_x_ptr, index=(col_block,), tile=acc_x)
    # bf16 round-trip on final grad sum (matches Triton)
    bf = ct.astype(acc_grad, ct.bfloat16)
    ct.store(out_grad_sum_ptr, index=(col_block,), tile=ct.astype(bf, ct.float32))


# 11a5c74c: hidden=768 (non-pow2). BLOCK_C=256 evenly divides 768.
@oracle_impl(hardware="B200", point="11a5c74c", BLOCK_C=256, NORM_FACTOR=768)
# 96a1854b: hidden=4096
@oracle_impl(hardware="B200", point="96a1854b", BLOCK_C=4096, NORM_FACTOR=768)
# 4676c06c: hidden=2048
@oracle_impl(hardware="B200", point="4676c06c", BLOCK_C=2048, NORM_FACTOR=768)
def oracle_forward(inputs, *, BLOCK_C: int, NORM_FACTOR: int):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs
    m = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    device = arg0_1.device

    num_col_blocks = c // BLOCK_C
    block_c_pad = _pow2_ceil(c)

    grad = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=device, dtype=torch.float32,
    )
    grad_2d = grad.view(m, c)
    grad_bf16 = torch.empty_strided(
        (m, c), (c, 1), device=device, dtype=torch.bfloat16,
    )

    row_sum = torch.empty((m,), device=device, dtype=torch.float32)
    row_dot = torch.empty((m,), device=device, dtype=torch.float32)
    partials = torch.empty((m, 3, c), device=device, dtype=torch.float32)

    sum_x_rhs = torch.empty((c,), device=device, dtype=torch.float32)
    sum_x = torch.empty((c,), device=device, dtype=torch.float32)
    sum_grad_bf16 = torch.empty((c,), device=device, dtype=torch.float32)

    # Flatten 3D inputs to 2D for cuTile
    rhs_2d = arg2_1.view(m, c)
    row_scale_1d = arg3_1.view(m)

    stream = torch.cuda.current_stream()
    # Kernel 1: row stats
    ct.launch(
        stream, (m, 1, 1), _row_stats_kernel,
        (arg0_1, arg1_1, rhs_2d, row_sum, row_dot, c, block_c_pad),
    )
    # Kernel 2: store + partial per (row, col_block)
    ct.launch(
        stream, (m, num_col_blocks, 1), _store_and_partial_kernel,
        (arg0_1, arg1_1, rhs_2d, row_scale_1d, row_sum, row_dot,
         grad_2d, grad_bf16, partials,
         c, NORM_FACTOR, BLOCK_C),
    )
    # Kernel 3: finalize column sums
    ct.launch(
        stream, (num_col_blocks, 1, 1), _finalize_channel_sums_kernel,
        (partials, sum_x_rhs, sum_x, sum_grad_bf16, m, BLOCK_C),
    )

    return grad, sum_x_rhs, sum_x, grad_bf16, grad_bf16.permute(1, 0), sum_grad_bf16
