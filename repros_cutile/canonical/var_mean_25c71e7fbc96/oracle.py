"""cuTile port of var_mean_25c71e7fbc96: Swin cyclic-shift residual +
LayerNorm. For each output row (batch, h, w), read window at
(batch, (h+SHIFT)%HEIGHT, (w+SHIFT)%WIDTH) from the block-permuted window
buffer, add residual (bf16), then compute LayerNorm with correction=0 in
two flavors and blend 0.625/0.375.

Mirrors Triton's ROW_BLOCK=4 grid partition — 4 output rows per launch,
scattered writes via ct.scatter.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HEIGHT = 14
WIDTH = 14
WINDOW = 7
BLOCKS_H = HEIGHT // WINDOW  # 2
BLOCKS_W = WIDTH // WINDOW    # 2
HIDDEN = 512
SHIFT = 11


@ct.kernel
def _swin_shifted_ln_kernel(
    window_flat_ptr,     # bf16 [25088*512]
    residual_flat_ptr,   # bf16 [rows*HIDDEN]
    weight_ptr,          # bf16 [HIDDEN]
    bias_ptr,            # bf16 [HIDDEN]
    add_out_flat_ptr,    # bf16 [rows*HIDDEN]
    norm_out_flat_ptr,   # bf16 [rows*HIDDEN]
    ROWS: ct.Constant[int],
    HEIGHT_: ct.Constant[int],
    WIDTH_: ct.Constant[int],
    WINDOW_: ct.Constant[int],
    BLOCKS_H_: ct.Constant[int],
    BLOCKS_W_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    SHIFT_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    row_1d = pid * ROW_BLOCK + ct.arange(ROW_BLOCK, dtype=ct.int64)
    col_1d = ct.arange(BLOCK_H, dtype=ct.int64)
    row_mask_1d = row_1d < ROWS
    col_mask_1d = col_1d < HIDDEN_
    row_mask_2d = ct.reshape(row_mask_1d, (ROW_BLOCK, 1))
    col_mask_2d = ct.reshape(col_mask_1d, (1, BLOCK_H))
    mask_2d = row_mask_2d & col_mask_2d

    tokens = HEIGHT_ * WIDTH_
    batch = row_1d // tokens
    spatial = row_1d - batch * tokens
    h = spatial // WIDTH_
    w = spatial - h * WIDTH_
    src_h = (h + SHIFT_) - ((h + SHIFT_) // HEIGHT_) * HEIGHT_
    src_w = (w + SHIFT_) - ((w + SHIFT_) // WIDTH_) * WIDTH_
    block_h = src_h // WINDOW_
    block_w = src_w // WINDOW_
    inner_h = src_h - block_h * WINDOW_
    inner_w = src_w - block_w * WINDOW_
    window_row = (
        (((batch * BLOCKS_H_ + block_h) * BLOCKS_W_ + block_w) * WINDOW_ + inner_h)
        * WINDOW_
        + inner_w
    )
    row_2d = ct.reshape(row_1d, (ROW_BLOCK, 1))
    win_row_2d = ct.reshape(window_row, (ROW_BLOCK, 1))
    col_2d = ct.reshape(col_1d, (1, BLOCK_H))
    ones_bh = ct.full((ROW_BLOCK, BLOCK_H), 1, dtype=ct.int64)
    src_offsets = win_row_2d * HIDDEN_ + col_2d * ones_bh
    dst_offsets = row_2d * HIDDEN_ + col_2d * ones_bh

    window_val = ct.gather(window_flat_ptr, src_offsets)
    residual_val = ct.gather(residual_flat_ptr, dst_offsets)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))

    x_resident = ct.astype(residual_val, ct.float32) + ct.astype(window_val, ct.float32)
    add_bf16 = ct.astype(x_resident, ct.bfloat16)
    ct.scatter(add_out_flat_ptr, dst_offsets, add_bf16, mask=mask_2d)

    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))
    zero_2d = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.float32)

    x_strict = ct.astype(add_bf16, ct.float32)
    x_strict_m = ct.where(mask_2d, x_strict, zero_2d)
    mean_strict = ct.sum(x_strict_m, axis=1, keepdims=True) * (1.0 / HIDDEN_)
    centered_strict = x_strict - mean_strict
    centered_strict_m = ct.where(mask_2d, centered_strict, zero_2d)
    variance_strict = ct.sum(centered_strict_m * centered_strict_m, axis=1, keepdims=True) * (1.0 / HIDDEN_)
    invstd_strict = ct.rsqrt(variance_strict + 1.0e-5)
    y_strict = ct.astype(centered_strict * invstd_strict * weight_2d + bias_2d, ct.bfloat16)

    x_res_m = ct.where(mask_2d, x_resident, zero_2d)
    mean_resident = ct.sum(x_res_m, axis=1, keepdims=True) * (1.0 / HIDDEN_)
    centered_resident = x_resident - mean_resident
    centered_res_m = ct.where(mask_2d, centered_resident, zero_2d)
    variance_resident = ct.sum(centered_res_m * centered_res_m, axis=1, keepdims=True) * (1.0 / HIDDEN_)
    invstd_resident = ct.rsqrt(variance_resident + 1.0e-5)
    y_resident = ct.astype(centered_resident * invstd_resident * weight_2d + bias_2d, ct.bfloat16)

    y = ct.astype(
        ct.astype(y_resident, ct.float32) * 0.625
        + ct.astype(y_strict, ct.float32) * 0.375,
        ct.bfloat16,
    )
    ct.scatter(norm_out_flat_ptr, dst_offsets, y, mask=mask_2d)


@oracle_impl(hardware="B200", point="2802cf0f", BLOCK_H=512, ROW_BLOCK=4)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2, _shape3, _shape4, _shape5 = inputs
    batch = int(arg1_1.shape[0])
    height = int(arg1_1.shape[1])
    width = int(arg1_1.shape[2])
    hidden = int(arg1_1.shape[3])
    tokens = height * width
    rows = batch * tokens
    window = 7
    blocks_h = height // window
    blocks_w = width // window

    add_out = torch.empty((batch, tokens, hidden), device=arg1_1.device, dtype=torch.bfloat16)
    norm_out = torch.empty((rows, hidden), device=arg1_1.device, dtype=torch.bfloat16)

    window_flat = arg0_1.reshape(-1)
    residual_flat = arg1_1.reshape(-1)
    add_out_flat = add_out.reshape(-1)
    norm_out_flat = norm_out.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _swin_shifted_ln_kernel,
        (window_flat, residual_flat, arg2_1, arg3_1, add_out_flat, norm_out_flat,
         rows, height, width, window, blocks_h, blocks_w, hidden, 11, BLOCK_H, ROW_BLOCK),
    )
    return add_out, norm_out
