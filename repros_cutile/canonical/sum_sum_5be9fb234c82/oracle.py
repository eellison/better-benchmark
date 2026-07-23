"""cuTile port of sum_sum_5be9fb234c82: MT5 dropout/RMSNorm backward.

For each row: compute the two RMSNorm backward tensors + column partial
sums. A finalize kernel then reduces the column partials over rows.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 1.1111111111111112


@ct.kernel
def _row_kernel(
    x_ptr,          # bf16 [M, D]
    mask1_ptr,      # bool [M, D]
    weight_ptr,     # f32 [D]
    arg_ptr,        # f32 [M, D]
    row_scale_ptr,  # f32 [M]
    mask2_ptr,      # bool [M, D]
    partial_col_ptr,  # f32 [M, D]  (one row of partial per row; a finalize kernel sums)
    add_out_ptr,    # f32 [M, D]
    bf16_out_ptr,   # bf16 [M, D]
    D: ct.Constant[int],
    INV_D: ct.Constant[float],
):
    row = ct.bid(0)
    x = ct.astype(ct.load(x_ptr, index=(row, 0), shape=(1, D)), ct.float32)
    mask1_bool = ct.load(mask1_ptr, index=(row, 0), shape=(1, D))
    weight = ct.load(weight_ptr, index=(0,), shape=(D,))
    weight_2d = ct.reshape(weight, (1, D))
    arg = ct.load(arg_ptr, index=(row, 0), shape=(1, D))
    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    row_scale_2d = ct.reshape(row_scale, (1, 1))
    mask2_bool = ct.load(mask2_ptr, index=(row, 0), shape=(1, D))

    mask1_f = ct.where(
        mask1_bool,
        ct.full(shape=(1, D), fill_value=1.0, dtype=ct.float32),
        ct.full(shape=(1, D), fill_value=0.0, dtype=ct.float32),
    )
    dropout_scale = mask1_f * SCALE
    masked_x = x * dropout_scale
    weighted = masked_x * weight_2d
    arg_scaled = arg * row_scale_2d
    # partial column sum: sum over row of masked_x * arg_scaled per column.
    # Since we're only processing one row, we output masked_x * arg_scaled directly.
    partial = masked_x * arg_scaled
    ct.store(partial_col_ptr, index=(row, 0), tile=partial)

    row_sum_values = weighted * arg  # (1, D)
    row_sum = ct.sum(row_sum_values, axis=1, keepdims=True)  # (1, 1)

    row_scale_sq = row_scale_2d * row_scale_2d
    row_scale_cu = row_scale_sq * row_scale_2d
    correction_scale = (row_sum * -0.5) * row_scale_cu
    correction_scale = correction_scale * INV_D
    correction = correction_scale * (arg * 2.0)
    direct = weighted * row_scale_2d
    add_value = direct + correction
    ct.store(add_out_ptr, index=(row, 0), tile=add_value)

    add_bf16 = ct.astype(add_value, ct.bfloat16)
    mask2_f = ct.where(
        mask2_bool,
        ct.full(shape=(1, D), fill_value=SCALE, dtype=ct.float32),
        ct.full(shape=(1, D), fill_value=0.0, dtype=ct.float32),
    )
    mask_scale_bf16 = ct.astype(mask2_f, ct.bfloat16)
    out_bf16 = ct.astype(
        ct.astype(add_bf16, ct.float32) * ct.astype(mask_scale_bf16, ct.float32),
        ct.bfloat16,
    )
    ct.store(bf16_out_ptr, index=(row, 0), tile=out_bf16)


@ct.kernel
def _finalize_kernel(
    partial_ptr,    # f32 [M, D]
    out_ptr,        # f32 [D]
    M: ct.Constant[int],
    D: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)
    acc = ct.zeros(shape=(BLOCK_COLS,), dtype=ct.float32)
    for r in ct.static_iter(range(M // BLOCK_ROWS)):
        chunk = ct.load(partial_ptr, index=(r, col_block), shape=(BLOCK_ROWS, BLOCK_COLS))
        acc = acc + ct.sum(chunk, axis=0)
    ct.store(out_ptr, index=(col_block,), tile=acc)


@oracle_impl(
    hardware="B200",
    point="8b6fe472",
    ROW_TILE=4,
    BLOCK_D=512,
    FINAL_BLOCK_D=8,
    FINAL_BLOCK_TILES=1024,
)
def oracle_forward(inputs, **_kwargs):
    (
        x, mask1, weight, arg, row_scale, mask2,
        _shape0, _shape1, _shape2, _shape3,
    ) = inputs

    m, d = (int(dim) for dim in x.shape)
    device = x.device

    partial_col_2d = torch.empty((m, d), device=device, dtype=torch.float32)
    col_out = torch.empty((d,), device=device, dtype=torch.float32)
    add_out = torch.empty((32, 128, d), device=device, dtype=torch.float32)
    bf16_out = torch.empty_strided(
        tuple(x.shape),
        tuple(int(s) for s in x.stride()),
        device=device,
        dtype=torch.bfloat16,
    )

    add_out_2d = add_out.view(m, d)
    mask1_2d = mask1.view(m, d)
    arg_2d = arg.view(m, d)
    row_scale_1d = row_scale.view(m)
    mask2_2d = mask2.view(m, d)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (m, 1, 1),
        _row_kernel,
        (x, mask1_2d, weight, arg_2d, row_scale_1d, mask2_2d,
         partial_col_2d, add_out_2d, bf16_out,
         d, 1.0 / d),
    )

    BLOCK_ROWS = 32
    BLOCK_COLS = 32
    while d % BLOCK_COLS != 0:
        BLOCK_COLS //= 2
    ct.launch(
        stream,
        (d // BLOCK_COLS, 1, 1),
        _finalize_kernel,
        (partial_col_2d, col_out, m, d, BLOCK_ROWS, BLOCK_COLS),
    )
    return col_out, add_out, bf16_out, bf16_out.permute(1, 0)
