"""cuTile port of sum_03788f40c42f: DistillGPT2/GPT2 tanh-GELU backward + column sum.

Materializes the bf16 GELU-backward output tensor via cuTile, then computes
the column sum with a second cuTile reduction kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


COLS = 3072
GELU_SCALE = 0.7978845608028654
GELU_COEFF = 0.044715


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _gelu_backward_kernel(
    grad_ptr,        # bf16 [rows, cols]
    x_ptr,           # bf16 [rows, cols]
    out_ptr,         # bf16 [rows, cols]
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)
    grad_bf = ct.load(grad_ptr, index=(row, col_block), shape=(1, BLOCK_C))
    x_bf = ct.load(x_ptr, index=(row, col_block), shape=(1, BLOCK_C))

    grad_f = ct.astype(grad_bf, ct.float32)
    x_f = ct.astype(x_bf, ct.float32)

    half_x_bf = ct.astype(x_f * 0.5, ct.bfloat16)
    mul_1 = grad_f * ct.astype(half_x_bf, ct.float32)
    x2 = x_f * x_f
    x3 = x2 * x_f
    tanh_arg = (x_f + x3 * GELU_COEFF) * GELU_SCALE
    # tanh via (e^x - e^-x)/(e^x + e^-x)
    e_pos = ct.exp(tanh_arg)
    e_neg = ct.exp(-tanh_arg)
    tanh_val = (e_pos - e_neg) / (e_pos + e_neg)
    mul_4 = grad_f * (tanh_val + 1.0)
    sub = 1.0 - tanh_val * tanh_val
    mul_7 = (mul_1 * sub) * GELU_SCALE
    convert_3 = ct.astype(mul_7, ct.bfloat16)
    mul_10 = (mul_7 * GELU_COEFF) * (x2 * 3.0)
    convert_4 = ct.astype(mul_10, ct.bfloat16)
    add_2 = ct.astype(ct.astype(convert_3, ct.float32) + ct.astype(convert_4, ct.float32), ct.bfloat16)
    convert_2 = ct.astype(mul_4, ct.bfloat16)
    mul_11 = ct.astype(ct.astype(convert_2, ct.float32) * 0.5, ct.bfloat16)
    out_bf = ct.astype(ct.astype(add_2, ct.float32) + ct.astype(mul_11, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(row, col_block), tile=out_bf)


@ct.kernel
def _col_sum_kernel(
    x_ptr,           # bf16 [rows, cols]
    sum_ptr,         # f32 [cols]
    ROWS: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    N_ROW_BLOCKS: ct.Constant[int],
):
    col_block = ct.bid(0)
    acc = ct.zeros((BLOCK_C,), dtype=ct.float32)
    for r_block in range(N_ROW_BLOCKS):
        x = ct.load(
            x_ptr, index=(r_block, col_block), shape=(BLOCK_R, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        # Mask OOB rows.
        rows_axis = ct.arange(BLOCK_R, dtype=ct.int32) + r_block * BLOCK_R
        valid = rows_axis < ROWS
        valid_2d = ct.reshape(valid, (BLOCK_R, 1))
        x_f = ct.astype(x, ct.float32)
        zero_f = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=ct.float32)
        x_masked = ct.where(valid_2d, x_f, zero_f)
        block_sum = ct.sum(x_masked, axis=0)
        acc = acc + block_sum
    # Round-trip through bf16.
    col_sum_bf = ct.astype(acc, ct.bfloat16)
    col_sum_f = ct.astype(col_sum_bf, ct.float32)
    ct.store(sum_ptr, index=(col_block,), tile=col_sum_f)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="05a5e903")
@oracle_impl(hardware="B200", point="58ff2bc5")
def oracle_forward(inputs):
    grad, x, _shape0, _shape1, out_shape_arg, sum_shape_arg = inputs
    rows = int(grad.shape[0])
    cols = int(grad.shape[1])
    device = grad.device

    BLOCK_C = 1024  # divides 3072 into 3 parts
    ROWS_PAD = _next_pow2(rows)  # 16384 or 8192

    out = torch.empty_strided(
        _shape_tuple(out_shape_arg), (cols, 1),
        device=device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(
        _shape_tuple(sum_shape_arg), (1,),
        device=device, dtype=torch.float32)

    # Reshape grad and x to (rows, cols) contiguous view.
    grad_2d = grad.view(rows, cols)
    x_2d = x.view(rows, cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, cols // BLOCK_C, 1), _gelu_backward_kernel,
        (grad_2d, x_2d, out, BLOCK_C),
    )
    # Reduce columns: iterate row-blocks internally.
    BLOCK_R = 256
    n_row_blocks = (rows + BLOCK_R - 1) // BLOCK_R
    ct.launch(
        stream, (cols // BLOCK_C, 1, 1), _col_sum_kernel,
        (out, sum_out, rows, BLOCK_R, BLOCK_C, n_row_blocks),
    )
    return out, sum_out
