"""cuTile port of var_mean_0b4143d6fc46: Whisper residual+LayerNorm.

For each row: bf16 residual add (strided residual), fp32 var+mean over HIDDEN,
rsqrt(eps=1e-5), affine bf16 store. HIDDEN=384 padded to BLOCK_H=512.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _whisper_residual_ln_kernel(
    flat_ptr,       # bf16 [ROWS, HIDDEN]
    residual_ptr,   # bf16 [ROWS, HIDDEN] (strided)
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    add_out_ptr,    # bf16 [ROWS, HIDDEN] (same strided layout)
    norm_out_ptr,   # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))

    flat = ct.astype(
        ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    residual = ct.astype(
        ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    added_bf16 = ct.astype(residual + flat, ct.bfloat16)
    # NOTE: store may write OOB for the padded region; but since BLOCK_H is
    # aligned to power of 2 and ROWS*HIDDEN in bf16 has enough allocated space
    # we can safely store BLOCK_H columns even if HIDDEN < BLOCK_H. If not, we
    # need a masked store — use scatter approach instead. Skip for now: use
    # separate loop with only HIDDEN columns.
    # For safety, avoid store past HIDDEN: reshape and use scatter.
    x = ct.astype(added_bf16, ct.float32)
    x_masked = ct.where(col_mask_2d, x, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    # Store — write only up to HIDDEN in the storage. Since BLOCK_H > HIDDEN,
    # we may overwrite adjacent memory. Use the fact that the storage is laid
    # out so that a full BLOCK_H write per row would collide with next row's
    # data. To avoid this, use scatter with mask.
    # scatter approach: use ct.scatter with column indices.
    col_idx = ct.arange(BLOCK_H, dtype=ct.int64)
    row_idx = ct.full(shape=(BLOCK_H,), fill_value=row, dtype=ct.int64)
    ct.scatter(add_out_ptr, (row_idx, col_idx), ct.reshape(added_bf16, (BLOCK_H,)), mask=col_mask)
    ct.scatter(norm_out_ptr, (row_idx, col_idx), ct.reshape(affine_bf16, (BLOCK_H,)), mask=col_mask)


@oracle_impl(hardware="B200", point="aafbb27e", BLOCK_H=512, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    add_shape = tuple(int(d) for d in shape0)
    view_shape_1 = tuple(int(d) for d in shape1)
    view_shape_2 = tuple(int(d) for d in shape2)
    view_shape_3 = tuple(int(d) for d in shape3)
    rows = int(view_shape_1[0])
    seq_len = int(add_shape[1])
    hidden = int(add_shape[2])

    add_out = torch.empty_strided(
        add_shape,
        (seq_len * hidden, 1, seq_len),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        add_shape,
        (seq_len * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # Views for the kernel: flat/norm are contiguous 2D; residual/add_out are strided 2D
    residual_2d = arg1_1.squeeze(0)  # (rows, hidden), stride=(1, seq_len)
    add_out_2d = add_out.squeeze(0)  # same stride pattern
    norm_out_2d = norm_base.squeeze(0)  # (rows, hidden), contiguous

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _whisper_residual_ln_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, add_out_2d, norm_out_2d, hidden, BLOCK_H),
    )
    return (
        add_out,
        norm_base.view(view_shape_1),
        norm_base.view(view_shape_2),
        norm_base.view(view_shape_3),
    )
