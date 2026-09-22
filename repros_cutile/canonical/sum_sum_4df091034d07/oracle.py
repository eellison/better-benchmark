"""cuTile port of sum_sum_4df091034d07: MT5 LayerNorm-backward tail fragment.

The Triton oracle fuses row-local reductions, a column reduction, a residual
add, and a bf16 dropout-scaled emit. Port structure:

* Kernel 1 (per row): compute the returned f32 tensor `add_2` and the bf16
  dropout-scaled output for one row. Emit a per-row partial column sum
  `add * arg3 * arg4` (float32).
* Kernel 2 (row-reduce partial columns): reduce partial column sums along
  the row-tile axis into the final `[D]` output.

The inline PTX round-to-nearest arithmetic is IEEE 754 RN by default in
cuTile, so those ops become normal `+ * -`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 4096
D = 512
DROPOUT_SCALE = 1.1111111111111112
INV_D = 0.001953125


@ct.kernel
def _mt5_row_kernel(
    arg0_ptr,      # bf16 [M, D]
    arg1_ptr,      # bf16 [M, D]
    weight_ptr,    # f32  [D]
    norm_arg_ptr,  # f32  [M, D]
    row_scale_ptr, # f32  [M]
    residual_ptr,  # f32  [M, D]
    mask_ptr,      # b8   [M, D]
    partial_col_ptr,  # f32 [M, D]  (partial column contribs)
    out_f32_ptr,   # f32  [M, D]
    out_bf16_ptr,  # bf16 [M, D]
    D_: ct.Constant[int],
    INV_D_: ct.Constant[int],
):
    row = ct.bid(0)

    x0 = ct.load(arg0_ptr, index=(row, 0), shape=(1, D_))
    x1 = ct.load(arg1_ptr, index=(row, 0), shape=(1, D_))
    weight = ct.load(weight_ptr, index=(0,), shape=(D_,))
    weight = ct.reshape(weight, (1, D_))

    norm_arg = ct.load(norm_arg_ptr, index=(row, 0), shape=(1, D_))
    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    row_scale = ct.reshape(row_scale, (1, 1))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, D_))

    x0_f = ct.astype(x0, ct.float32)
    x1_f = ct.astype(x1, ct.float32)
    add_value = x0_f + x1_f

    norm_scaled = norm_arg * row_scale
    partial_val = add_value * norm_scaled
    ct.store(partial_col_ptr, index=(row, 0), tile=partial_val)

    weighted = add_value * weight
    row_terms = weighted * norm_arg
    row_sum = ct.sum(row_terms)  # scalar

    direct = weighted * row_scale
    add_1 = residual + direct

    scale_sq = row_scale * row_scale
    scale_cu = scale_sq * row_scale
    mul_5 = row_sum * (-0.5)
    mul_6 = mul_5 * scale_cu
    div = mul_6 * INV_D_
    mul_7 = norm_arg * 2.0
    mul_8 = div * mul_7
    add_2 = add_1 + mul_8
    ct.store(out_f32_ptr, index=(row, 0), tile=add_2)

    mask_val = ct.load(mask_ptr, index=(row, 0), shape=(1, D_))
    mask_bf = ct.astype(mask_val, ct.bfloat16)
    mask_scaled = ct.astype(ct.astype(mask_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_2_bf16 = ct.astype(add_2, ct.bfloat16)
    prod_f = ct.astype(add_2_bf16, ct.float32) * ct.astype(mask_scaled, ct.float32)
    out_bf16 = ct.astype(prod_f, ct.bfloat16)
    ct.store(out_bf16_ptr, index=(row, 0), tile=out_bf16)


@ct.kernel
def _col_sum_kernel(
    partial_ptr,   # f32 [M, D]
    col_sum_ptr,   # f32 [D]
    M_: ct.Constant[int],
    D_: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Sum M rows across a (BLOCK_M x BLOCK_D) tile.
    # BLOCK_M chosen so BLOCK_M divides M.
    acc = ct.zeros((BLOCK_D,), dtype=ct.float32)
    m_iters = M_ // BLOCK_M
    for m_i in range(m_iters):
        vals = ct.load(partial_ptr, index=(m_i, col_block), shape=(BLOCK_M, BLOCK_D))
        acc = acc + ct.sum(vals, axis=0)
    ct.store(col_sum_ptr, index=(col_block,), tile=acc)


@oracle_impl(hardware="B200", point="c98bcb56", ROW_TILE=4, FINAL_BLOCK_D=8)
def oracle_forward(inputs, *, ROW_TILE, FINAL_BLOCK_D):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    ) = inputs

    device = arg0_1.device
    # Reshape 3D f32 inputs to 2D for the kernel.
    arg3_2d = arg3_1.view(M, D)
    arg4_1d = arg4_1.view(M)
    arg5_2d = arg5_1.view(M, D)
    arg6_2d = arg6_1.view(M, D)

    out_col = torch.empty((D,), device=device, dtype=torch.float32)
    out_f32_2d = torch.empty((M, D), device=device, dtype=torch.float32)
    out_bf16 = torch.empty((M, D), device=device, dtype=torch.bfloat16)
    partial = torch.empty((M, D), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (M, 1, 1),
        _mt5_row_kernel,
        (
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_2d,
            arg4_1d,
            arg5_2d,
            arg6_2d,
            partial,
            out_f32_2d,
            out_bf16,
            D,
            INV_D,
        ),
    )

    # Column-reduce partial [M, D] -> [D]
    BLOCK_M = 64
    BLOCK_D = 32
    ct.launch(
        stream,
        (D // BLOCK_D, 1, 1),
        _col_sum_kernel,
        (partial, out_col, M, D, BLOCK_M, BLOCK_D),
    )

    out_f32 = out_f32_2d.view(32, 128, D)
    return out_col, out_f32, out_bf16, out_bf16.permute(1, 0)
