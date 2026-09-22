"""cuTile port of var_mean_var_mean_0ec03ae7565f: Swin dual channel-LayerNorm
plus window partition.

Input bf16 [BATCH, CHANNELS, HEIGHT, WIDTH] (strided); output tensors:
- first_out: bf16 [BATCH, HEIGHT, WIDTH, CHANNELS]
- window_out: bf16 [BATCH*GRID_H*GRID_W*WINDOW_H*WINDOW_W, CHANNELS]

Two chained LayerNorms over channel dim (128), then a 7x7 window partition
mapping to output row = ((batch*GRID_H + wr)*GRID_W + wc)*WH*WW + inner_h*WW + inner_w.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 128
HEIGHT = 56
WIDTH = 56
GRID_H = 8
GRID_W = 8
WINDOW_H = 7
WINDOW_W = 7
ROWS = BATCH * HEIGHT * WIDTH  # 128*56*56 = 401408


@ct.kernel
def _swin_dual_ln_kernel(
    x_ptr,           # bf16 [BATCH, HEIGHT, WIDTH, CHANNELS] contiguous (view after permute)
    weight1_ptr,     # bf16 [CHANNELS]
    bias1_ptr,       # bf16 [CHANNELS]
    weight2_ptr,     # bf16 [CHANNELS]
    bias2_ptr,       # bf16 [CHANNELS]
    first_out_ptr,   # bf16 [ROWS, CHANNELS]
    window_out_ptr,  # bf16 [ROWS, CHANNELS]  (rows already reordered)
    ROWS_C: ct.Constant[int],
    CHANNELS_C: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    EPS_INV_C: ct.Constant[float],
):
    # x_ptr is a 2D array of shape [ROWS, CHANNELS] in the permuted layout.
    row_block = ct.bid(0)
    x = ct.load(x_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_C))
    x_f = ct.astype(x, ct.float32)

    inv_c = 1.0 / CHANNELS_C
    mean1 = ct.sum(x_f, axis=1, keepdims=True) * inv_c
    centered1 = x_f - mean1
    var1 = ct.sum(centered1 * centered1, axis=1, keepdims=True) * inv_c
    invstd1 = ct.rsqrt(var1 + 1.0e-5)

    w1 = ct.load(weight1_ptr, index=(0,), shape=(BLOCK_C,))
    b1 = ct.load(bias1_ptr, index=(0,), shape=(BLOCK_C,))
    w1_f = ct.astype(ct.reshape(w1, (1, BLOCK_C)), ct.float32)
    b1_f = ct.astype(ct.reshape(b1, (1, BLOCK_C)), ct.float32)

    normalized1 = centered1 * invstd1
    y1_f = normalized1 * w1_f + b1_f
    y1_bf = ct.astype(y1_f, ct.bfloat16)

    # Store first_out (permute already applied — first_out is [ROWS, CHANNELS])
    ct.store(first_out_ptr, index=(row_block, 0), tile=y1_bf)

    # Second LN over the rounded y1
    y1_f2 = ct.astype(y1_bf, ct.float32)
    mean2 = ct.sum(y1_f2, axis=1, keepdims=True) * inv_c
    centered2 = y1_f2 - mean2
    var2 = ct.sum(centered2 * centered2, axis=1, keepdims=True) * inv_c
    invstd2 = ct.rsqrt(var2 + 1.0e-5)

    w2 = ct.load(weight2_ptr, index=(0,), shape=(BLOCK_C,))
    b2 = ct.load(bias2_ptr, index=(0,), shape=(BLOCK_C,))
    w2_f = ct.astype(ct.reshape(w2, (1, BLOCK_C)), ct.float32)
    b2_f = ct.astype(ct.reshape(b2, (1, BLOCK_C)), ct.float32)

    normalized2 = centered2 * invstd2
    y2_f = normalized2 * w2_f + b2_f
    y2_bf = ct.astype(y2_f, ct.bfloat16)

    # window_out_ptr has already been laid out so that window rows are
    # contiguous in the second dim. We store y2_bf into window_out at
    # the correct row range, which is computed by the harness pre-permute.
    ct.store(window_out_ptr, index=(row_block, 0), tile=y2_bf)


@oracle_impl(hardware="B200", point="b68c1040", ROW_BLOCK=32, BLOCK_C=128)
def oracle_forward(inputs, *, ROW_BLOCK: int, BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape1, _shape2, _shape3 = inputs

    device = arg0_1.device
    # arg0_1 shape [128,128,56,56] channels-first — we need [128,56,56,128] channels-last flat
    x_perm = arg0_1.permute(0, 2, 3, 1).contiguous()  # (128,56,56,128)
    x_flat = x_perm.view(ROWS, CHANNELS)  # [ROWS, 128]

    first_out_flat = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.bfloat16)

    # We need window_out laid out as if each source row maps to a permuted
    # row. Rather than a permuted store index inside the kernel (which needs
    # extra indirection), we can store to a plain [ROWS, CHANNELS] buffer
    # and permute *outside* the kernel. But since the second store is just
    # y2 to a permuted layout, doing it via a temporary buffer + torch
    # gather/scatter is simplest.
    y2_flat = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(ROWS, ROW_BLOCK), 1, 1),
        _swin_dual_ln_kernel,
        (x_flat, arg1_1, arg2_1, arg3_1, arg4_1, first_out_flat, y2_flat,
         ROWS, CHANNELS, ROW_BLOCK, BLOCK_C, 1.0e-5),
    )

    # first_out reshaped to [BATCH, HEIGHT, WIDTH, CHANNELS]
    first_out = first_out_flat.view(BATCH, HEIGHT, WIDTH, CHANNELS)

    # window_out reshape: y2_flat is arranged as (batch, h, w, c). We need the
    # Swin window-partition ordering: view -> permute -> view -> ... ->
    # [ROWS, CHANNELS] where the last dim is CHANNELS.
    # Equivalent torch ops (from the Repro):
    # view [B, GH, WH, GW, WW, C] -> permute [0,1,3,2,4,5] -> clone -> view [8192,7,7,C] -> view [8192,49,C] -> view [401408, C]
    y2_bhwc = y2_flat.view(BATCH, HEIGHT, WIDTH, CHANNELS)
    y2_view = y2_bhwc.view(BATCH, GRID_H, WINDOW_H, GRID_W, WINDOW_W, CHANNELS)
    y2_perm = y2_view.permute(0, 1, 3, 2, 4, 5).contiguous()
    window_out = y2_perm.view(ROWS, CHANNELS)

    return first_out, window_out
