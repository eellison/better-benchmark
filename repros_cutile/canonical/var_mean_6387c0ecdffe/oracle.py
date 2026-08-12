"""cuTile port of var_mean_6387c0ecdffe (NEW_PATTERN): Swin window
LayerNorm + 7x7 window partition (no cyclic shift).

Fuses window partition with LayerNorm into a single kernel using
load_advanced_indexing to compute the source row per output row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _swin_ln_window_kernel(
    x_ptr,           # bf16 [ROWS, HIDDEN] source
    weight_ptr,      # bf16 [HIDDEN]
    bias_ptr,        # bf16 [HIDDEN]
    out_ptr,         # bf16 [ROWS, HIDDEN] window-permuted output
    HEIGHT: ct.Constant[int],
    WIDTH: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    WINDOW_H: ct.Constant[int],
    WINDOW_W: ct.Constant[int],
    GRID_H: ct.Constant[int],
    GRID_W: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    # Decode the window-partitioned output row -> source row.
    inner_w = row % WINDOW_W
    tmp = row // WINDOW_W
    inner_h = tmp % WINDOW_H
    tmp = tmp // WINDOW_H
    window_col = tmp % GRID_W
    tmp = tmp // GRID_W
    window_row = tmp % GRID_H
    batch_idx = tmp // GRID_H

    src_h = window_row * WINDOW_H + inner_h
    src_w = window_col * WINDOW_W + inner_w
    src_row = batch_idx * HEIGHT * WIDTH + src_h * WIDTH + src_w

    x_bf = ct.load(x_ptr, index=(src_row, 0), shape=(1, BLOCK_H))
    x = ct.astype(x_bf, ct.float32)
    inv_h = 1.0 / HIDDEN
    x_sum = ct.sum(x, axis=1, keepdims=True)
    mean = x_sum * inv_h
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = centered * invstd * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _shape_tuple(value):
    return tuple(int(dim) for dim in value)


@oracle_impl(hardware="B200", point="6d53c060")
@oracle_impl(hardware="B200", point="b408f6c8")
def oracle_forward(inputs):
    x, weight, bias, shape0, shape1, _shape2, _shape3, shape4 = inputs
    shape0 = _shape_tuple(shape0)  # [128, 14, 14, 512] or [128, 28, 28, 256]
    shape1 = _shape_tuple(shape1)  # [128, gh, wh, gw, ww, hidden]
    shape4 = _shape_tuple(shape4)  # [rows, hidden]

    batch, height, width, hidden = shape0
    _batch, grid_h, window_h, grid_w, window_w, _hidden = shape1
    rows = shape4[0]

    view = x.view(shape0)

    out = torch.empty((rows, hidden), device=x.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _swin_ln_window_kernel,
        (x, weight, bias, out,
         height, width, hidden, window_h, window_w, grid_h, grid_w, hidden),
    )
    return view, out
