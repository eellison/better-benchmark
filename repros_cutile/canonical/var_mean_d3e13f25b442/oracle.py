"""cuTile port of var_mean_d3e13f25b442 (SCHEDULER_FUSION): Swin shifted-window
residual LayerNorm.

Fuses residual-add + LayerNorm + shifted-window partition into one kernel
matching the Triton structure.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
HEIGHT = 14
WIDTH = 14
HIDDEN = 512
WINDOW = 7
SHIFT = 3
ROWS = BATCH * HEIGHT * WIDTH  # 25088


@ct.kernel
def _swin_shifted_window_ln_kernel(
    flat_ptr,       # bf16 [ROWS, HIDDEN] source
    residual_ptr,   # bf16 [ROWS, HIDDEN] source
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    add_out_ptr,    # bf16 [ROWS, HIDDEN]  (stored at source row)
    norm_out_ptr,   # bf16 [ROWS, HIDDEN]  (stored at window-permuted row)
    HEIGHT_C: ct.Constant[int],
    WIDTH_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    WINDOW_C: ct.Constant[int],
    SHIFT_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    out_row = ct.bid(0)
    # Decode window-partitioned output row -> shifted grid coord.
    inner_w = out_row % WINDOW_C
    tmp = out_row // WINDOW_C
    inner_h = tmp % WINDOW_C
    tmp = tmp // WINDOW_C
    window_col = tmp % (WIDTH_C // WINDOW_C)
    tmp = tmp // (WIDTH_C // WINDOW_C)
    window_row = tmp % (HEIGHT_C // WINDOW_C)
    batch = tmp // (HEIGHT_C // WINDOW_C)

    shifted_h = window_row * WINDOW_C + inner_h
    shifted_w = window_col * WINDOW_C + inner_w
    src_h = (shifted_h + SHIFT_C) % HEIGHT_C
    src_w = (shifted_w + SHIFT_C) % WIDTH_C
    src_row = batch * (HEIGHT_C * WIDTH_C) + src_h * WIDTH_C + src_w

    flat_bf = ct.load(flat_ptr, index=(src_row, 0), shape=(1, BLOCK_H))
    resid_bf = ct.load(residual_ptr, index=(src_row, 0), shape=(1, BLOCK_H))
    flat_f = ct.astype(flat_bf, ct.float32)
    resid_f = ct.astype(resid_bf, ct.float32)
    add_bf = ct.astype(resid_f + flat_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(src_row, 0), tile=add_bf)

    x = ct.astype(add_bf, ct.float32)
    inv_h = 1.0 / HIDDEN_C
    mean = ct.sum(x, axis=1, keepdims=True) * inv_h
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    y = centered * invstd * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(out_row, 0), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="1ae3d509", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _s0, _s1, _s2, _s3, _s4, _s5 = inputs
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
    add_out_2d = add_out.view(ROWS, HIDDEN)
    residual_2d = arg1_1.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _swin_shifted_window_ln_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, add_out_2d, norm_out,
         HEIGHT, WIDTH, HIDDEN, WINDOW, SHIFT, BLOCK_H),
    )
    return add_out, norm_out
