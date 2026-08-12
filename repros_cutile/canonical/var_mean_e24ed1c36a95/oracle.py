"""cuTile port of var_mean_e24ed1c36a95: Swin window-reverse + cyclic-shift +
residual add + LayerNorm.

Approach: the window-reverse + cyclic-shift is a fixed data-independent
permutation on the input. We build a per-row source-index tensor on the host
(via torch) and use it to gather rows in the LayerNorm kernel. Everything
else is a per-row LayerNorm over hidden=128.

The Triton oracle's inline PTX fp32 arithmetic is IEEE-RN by default in
cuTile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
HEIGHT = 56
WIDTH = 56
HIDDEN = 128
WINDOW = 7
SHIFT = 53
TOKENS = HEIGHT * WIDTH  # 3136
ROWS = BATCH * TOKENS
BLOCKS_H = HEIGHT // WINDOW
BLOCKS_W = WIDTH // WINDOW


@ct.kernel
def _swin_ln_row_kernel(
    window_rows_ptr,     # bf16 [ROWS, HIDDEN]  (gathered pre-add)
    residual_flat_ptr,   # bf16 [ROWS, HIDDEN]
    weight_ptr,          # bf16 [HIDDEN]
    bias_ptr,            # bf16 [HIDDEN]
    add_out_ptr,         # bf16 [ROWS, HIDDEN]
    norm_out_ptr,        # bf16 [ROWS, HIDDEN]
    HIDDEN_C: ct.Constant[int],
):
    row = ct.bid(0)
    window_vals = ct.load(window_rows_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    residual_vals = ct.load(residual_flat_ptr, index=(row, 0), shape=(1, HIDDEN_C))

    win_f = ct.astype(window_vals, ct.float32)
    res_f = ct.astype(residual_vals, ct.float32)
    add_f = res_f + win_f
    add_bf16 = ct.astype(add_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=add_bf16)

    x = ct.astype(add_bf16, ct.float32)
    sum_x = ct.sum(x)
    mean = sum_x * (1.0 / HIDDEN_C)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(variance + 1.0e-5)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_C,))
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_C,))
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, HIDDEN_C))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, HIDDEN_C))
    y = normalized * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="8abb13ef")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2, _shape3, shape4, shape5 = inputs

    add_shape = tuple(int(dim) if int(dim) > 0 else TOKENS for dim in shape4)
    norm_shape = tuple(int(dim) for dim in shape5)
    device = arg1_1.device

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=device,
        dtype=torch.bfloat16,
    )

    # The Triton kernel computes window-reverse + cyclic shift as a
    # data-independent permutation: for each destination row (batch, h, w),
    # src_h = (h + 53) % 56, src_w = (w + 53) % 56, then re-window into
    # (block_h, block_w, inner_h, inner_w) coords within the [BATCH,
    # 8*8, 7*7, HIDDEN] input.
    h_range = torch.arange(HEIGHT, device=device)
    w_range = torch.arange(WIDTH, device=device)
    src_h = (h_range + SHIFT) % HEIGHT
    src_w = (w_range + SHIFT) % WIDTH
    block_h = src_h // WINDOW
    block_w = src_w // WINDOW
    inner_h = src_h - block_h * WINDOW
    inner_w = src_w - block_w * WINDOW
    win_row_local = (
        (block_h[:, None] * BLOCKS_W + block_w[None, :]) * WINDOW * WINDOW
        + inner_h[:, None] * WINDOW
        + inner_w[None, :]
    ).view(-1)  # [3136]

    window_bf = arg0_1.view(BATCH, TOKENS, HIDDEN)
    gathered_2d = window_bf.index_select(1, win_row_local).reshape(ROWS, HIDDEN)

    residual_flat = arg1_1.view(ROWS, HIDDEN)
    add_out_2d = add_out.view(ROWS, HIDDEN)
    norm_out_2d = norm_out.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _swin_ln_row_kernel,
        (
            gathered_2d,
            residual_flat,
            arg2_1,
            arg3_1,
            add_out_2d,
            norm_out_2d,
            HIDDEN,
        ),
    )
    return add_out, norm_out
