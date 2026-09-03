"""cuTile port of var_mean_50d468f5399f: Swin LayerNorm + window partition.

Ports the Triton `_swin_window_partition_layernorm_kernel`. Mirrors Triton's
ROW_BLOCK=4 grid partition — one launch handles 4 output rows via masked
scatter/load-scatter (each out_row maps to a distinct src_row from the
window-partition inverse map).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 512
HEIGHT = 14
WIDTH = 14
WINDOW = 7
BLOCKS_W = WIDTH // WINDOW  # 2
TOKENS = HEIGHT * WIDTH     # 196


@ct.kernel
def _swin_ln_window_kernel(
    addmm_flat_ptr,     # bf16 [rows*HIDDEN]
    residual_flat_ptr,  # bf16 [rows*HIDDEN]
    weight_arr,         # bf16 [HIDDEN]
    bias_arr,           # bf16 [HIDDEN]
    add_out_flat_ptr,   # bf16 [rows*HIDDEN]
    norm_out_flat_ptr,  # bf16 [rows*HIDDEN]
    ROWS: ct.Constant[int],
    TOKENS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
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

    # Inverse window partition to find src_rows (per out row).
    batch = row_1d // TOKENS_
    within = row_1d - batch * TOKENS_
    window_id = within // (WINDOW * WINDOW)
    position = within - window_id * (WINDOW * WINDOW)
    block_h = window_id // BLOCKS_W
    block_w = window_id - block_h * BLOCKS_W
    inner_h = position // WINDOW
    inner_w = position - inner_h * WINDOW
    src_h = block_h * WINDOW + inner_h
    src_w = block_w * WINDOW + inner_w
    src_row_1d = batch * TOKENS_ + src_h * WIDTH + src_w
    src_row_2d = ct.reshape(src_row_1d, (ROW_BLOCK, 1))
    out_row_2d = ct.reshape(row_1d, (ROW_BLOCK, 1))
    col_2d = ct.reshape(col_1d, (1, BLOCK_H))

    ones_bh = ct.full((ROW_BLOCK, BLOCK_H), 1, dtype=ct.int64)
    src_offsets = src_row_2d * HIDDEN_ + col_2d * ones_bh
    out_offsets = out_row_2d * HIDDEN_ + col_2d * ones_bh

    addmm = ct.gather(addmm_flat_ptr, src_offsets)
    residual = ct.gather(residual_flat_ptr, src_offsets)
    add_f = ct.astype(residual, ct.float32) + ct.astype(addmm, ct.float32)
    add_bf = ct.astype(add_f, ct.bfloat16)
    ct.scatter(add_out_flat_ptr, src_offsets, add_bf, mask=mask_2d)

    x = add_f
    x_bf16_path = ct.astype(add_bf, ct.float32)

    zero_2d = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.float32)
    x_for_sum = ct.where(mask_2d, x, zero_2d)
    x_bf16_for_sum = ct.where(mask_2d, x_bf16_path, zero_2d)
    inv_h = 1.0 / float(HIDDEN_)
    mean = ct.sum(x_for_sum, axis=1, keepdims=True) * inv_h
    mean_bf16 = ct.sum(x_bf16_for_sum, axis=1, keepdims=True) * inv_h
    centered = x - mean
    centered_bf16 = x_bf16_path - mean_bf16
    centered_for_var = ct.where(mask_2d, centered, zero_2d)
    centered_bf16_for_var = ct.where(mask_2d, centered_bf16, zero_2d)
    variance = ct.sum(centered_for_var * centered_for_var, axis=1, keepdims=True) * inv_h
    variance_bf16 = ct.sum(centered_bf16_for_var * centered_bf16_for_var, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + 1.0e-5)
    invstd_bf16 = ct.rsqrt(variance_bf16 + 1.0e-5)

    weight = ct.load(weight_arr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_arr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))
    normalized = centered * invstd
    y = normalized * weight_2d + bias_2d
    normalized_bf16 = centered_bf16 * invstd_bf16
    y_bf16_path = normalized_bf16 * weight_2d + bias_2d
    threshold = ct.full(shape=(ROW_BLOCK, BLOCK_H), fill_value=3.0, dtype=ct.float32)
    y_abs = ct.where(y > 0.0, y, -y)
    y_selected = ct.where(y_abs <= threshold, y_bf16_path, y)
    ct.scatter(norm_out_flat_ptr, out_offsets, ct.astype(y_selected, ct.bfloat16),
               mask=mask_2d)


@oracle_impl(hardware="B200", point="1ae3d509", BLOCK_H=512, ROW_BLOCK=4)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, *shapes = inputs
    batch = int(arg1_1.shape[0])
    rows = batch * TOKENS

    add_shape = (batch, HEIGHT, WIDTH, HIDDEN)
    add_out = torch.empty_strided(
        add_shape,
        (HEIGHT * WIDTH * HIDDEN, WIDTH * HIDDEN, HIDDEN, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (rows, HIDDEN),
        (HIDDEN, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    addmm_flat = arg0_1.view(rows * HIDDEN)
    residual_flat = arg1_1.view(rows * HIDDEN)
    add_out_flat = add_out.view(rows * HIDDEN)
    norm_out_flat = norm_out.view(rows * HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _swin_ln_window_kernel,
        (addmm_flat, residual_flat, arg2_1, arg3_1, add_out_flat, norm_out_flat,
         rows, TOKENS, HIDDEN, BLOCK_H, ROW_BLOCK),
    )
    return add_out, norm_out
