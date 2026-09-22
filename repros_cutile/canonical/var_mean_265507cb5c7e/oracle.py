"""cuTile port of var_mean_265507cb5c7e: Swin shifted-window residual LayerNorm.

Fused single-kernel port matching the Triton counterpart. For each shifted-
window row: gather flat+residual from src_row, add, write add_out at src_row,
LayerNorm on the sum, write norm_out at out_row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
HEIGHT = 28
WIDTH = 28
HIDDEN = 256
WINDOW = 7
SHIFT = 3
ROWS = BATCH * HEIGHT * WIDTH


@ct.kernel
def _swin_shifted_window_layernorm_kernel(
    flat_ptr,        # bf16 flat len ROWS*HIDDEN
    residual_ptr,    # bf16 flat len ROWS*HIDDEN
    weight_ptr,      # bf16 [HIDDEN]
    bias_ptr,        # bf16 [HIDDEN]
    add_out_ptr,     # bf16 flat len ROWS*HIDDEN
    norm_out_ptr,    # bf16 [ROWS, HIDDEN]
    ROWS_N: ct.Constant[int],
    HEIGHT_N: ct.Constant[int],
    WIDTH_N: ct.Constant[int],
    HIDDEN_N: ct.Constant[int],
    WINDOW_N: ct.Constant[int],
    SHIFT_N: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    block = ct.bid(0)
    out_rows = ct.arange(ROW_BLOCK, dtype=ct.int32) + block * ROW_BLOCK

    inner_w = out_rows - (out_rows // WINDOW_N) * WINDOW_N
    tmp = out_rows // WINDOW_N
    inner_h = tmp - (tmp // WINDOW_N) * WINDOW_N
    tmp2 = tmp // WINDOW_N
    wc_div = WIDTH_N // WINDOW_N
    window_col = tmp2 - (tmp2 // wc_div) * wc_div
    tmp3 = tmp2 // wc_div
    hr_div = HEIGHT_N // WINDOW_N
    window_row = tmp3 - (tmp3 // hr_div) * hr_div
    batch = tmp3 // hr_div

    shifted_h = window_row * WINDOW_N + inner_h + SHIFT_N
    src_h = shifted_h - (shifted_h // HEIGHT_N) * HEIGHT_N
    shifted_w = window_col * WINDOW_N + inner_w + SHIFT_N
    src_w = shifted_w - (shifted_w // WIDTH_N) * WIDTH_N
    src_spatial = src_h * WIDTH_N + src_w
    src_row = batch * (HEIGHT_N * WIDTH_N) + src_spatial

    rows_2d = ct.reshape(src_row, (ROW_BLOCK, 1))
    cols_1d = ct.arange(BLOCK_H, dtype=ct.int32)
    cols_2d = ct.reshape(cols_1d, (1, BLOCK_H))
    rows_bc = ct.broadcast_to(rows_2d, (ROW_BLOCK, BLOCK_H))
    cols_bc = ct.broadcast_to(cols_2d, (ROW_BLOCK, BLOCK_H))

    flat_vals = ct.astype(ct.gather(flat_ptr, (rows_bc, cols_bc)), ct.float32)
    resid_vals = ct.astype(ct.gather(residual_ptr, (rows_bc, cols_bc)), ct.float32)

    add_f = resid_vals + flat_vals
    add_bf = ct.astype(add_f, ct.bfloat16)
    ct.scatter(add_out_ptr, (rows_bc, cols_bc), add_bf)

    x = ct.astype(add_bf, ct.float32)
    mean = ct.sum(x, axis=1, keepdims=True) * (1.0 / HIDDEN_N)
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * (1.0 / HIDDEN_N)
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    w_f = ct.astype(weight, ct.float32)
    b_f = ct.astype(bias, ct.float32)
    w_2d = ct.reshape(w_f, (1, BLOCK_H))
    b_2d = ct.reshape(b_f, (1, BLOCK_H))
    normalized = centered * invstd
    out = normalized * w_2d + b_2d
    out_bf = ct.astype(out, ct.bfloat16)

    ct.store(norm_out_ptr, index=(block, 0), tile=out_bf)


@oracle_impl(hardware="B200", point="f4a631aa", BLOCK_H=256, ROW_BLOCK=8)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, *_ = inputs

    add_out = torch.empty_strided(
        (BATCH, HEIGHT, WIDTH, HIDDEN),
        (HEIGHT * WIDTH * HIDDEN, WIDTH * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    flat_2d = arg0_1  # already [ROWS, HIDDEN]
    residual_2d = arg1_1.view(ROWS, HIDDEN)
    add_out_2d = add_out.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(ROWS, ROW_BLOCK), 1, 1),
        _swin_shifted_window_layernorm_kernel,
        (flat_2d, residual_2d, arg2_1, arg3_1, add_out_2d, norm_out,
         ROWS, HEIGHT, WIDTH, HIDDEN, WINDOW, SHIFT, BLOCK_H, ROW_BLOCK),
    )
    return add_out, norm_out
