"""cuTile port of sum_sum_sum_be519a2435f2: FNet LayerNorm-backward + dropout.

Per-row: compute (weight*x*768 - sum(weight*x) - rhs*sum(weight*x*rhs)) * scale
Store as grad; multiply by dropout mask -> masked; then aggregate 3 column sums.

M=16384, C=768. C is not power of 2, pad with BLOCK_C=1024 and mask reductions.
Two kernels: per-row main + column-reduce over the emitted grad/x*rhs/x/masked.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 16384
C = 768
DROP_SCALE = 1.1111111111111112


@ct.kernel
def _row_ln_backward_kernel(
    x_ptr,          # f32 [M, C]
    weight_ptr,     # f32 [C]
    rhs_ptr,        # f32 [M, C]
    row_scale_ptr,  # f32 [M]
    drop_mask_ptr,  # bool [M, C]
    grad_out_ptr,   # f32 [M, BLOCK_C] padded
    masked_out_ptr, # f32 [M, BLOCK_C] padded
    partial_x_rhs_ptr,   # f32 [M, BLOCK_C] padded (per-row partial)
    partial_x_ptr,       # f32 [M, BLOCK_C] padded
    partial_masked_ptr,  # f32 [M, BLOCK_C] padded
    C_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    DROP_SCALE_C: ct.Constant[float],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_C),
                  padding_mode=ct.PaddingMode.ZERO)
    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    drop_mask = ct.load(drop_mask_ptr, index=(row, 0), shape=(1, BLOCK_C),
                        padding_mode=ct.PaddingMode.ZERO)

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = cols < C_C
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_C))
    zero = ct.full((1, BLOCK_C), 0.0, dtype=ct.float32)

    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    weighted = x * weight_2d
    weighted_masked = ct.where(col_mask_2d, weighted, zero)
    row_sum = ct.sum(weighted_masked)  # scalar

    weighted_rhs = weighted * rhs
    weighted_rhs_masked = ct.where(col_mask_2d, weighted_rhs, zero)
    row_dot = ct.sum(weighted_rhs_masked)  # scalar

    centered = weighted * float(C_C) - row_sum - rhs * row_dot
    row_scale_2d = ct.reshape(row_scale, (1, 1))
    grad = row_scale_2d * centered

    drop_f32 = ct.astype(drop_mask, ct.float32)
    mask_scaled = drop_f32 * DROP_SCALE_C
    masked = grad * mask_scaled

    ct.store(grad_out_ptr, index=(row, 0), tile=grad)
    ct.store(masked_out_ptr, index=(row, 0), tile=masked)

    # Store per-row partials (padded with zeros for masked-out columns).
    x_rhs = ct.where(col_mask_2d, x * rhs, zero)
    x_masked = ct.where(col_mask_2d, x, zero)
    masked_masked = ct.where(col_mask_2d, masked, zero)
    ct.store(partial_x_rhs_ptr, index=(row, 0), tile=x_rhs)
    ct.store(partial_x_ptr, index=(row, 0), tile=x_masked)
    ct.store(partial_masked_ptr, index=(row, 0), tile=masked_masked)


@ct.kernel
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,   # f32 [M, BLOCK_C_IN] (padded)
    partial_x_ptr,       # f32 [M, BLOCK_C_IN]
    partial_masked_ptr,  # f32 [M, BLOCK_C_IN]
    out_x_rhs_ptr,       # f32 [BLOCK_C_IN] (only cols 0..C_C valid)
    out_x_ptr,           # f32 [BLOCK_C_IN]
    out_masked_ptr,      # f32 [BLOCK_C_IN]
    M_C: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_C_IN: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
    FINAL_BLOCK_TILES: ct.Constant[int],
):
    col_tile = ct.bid(0)

    acc_x_rhs = ct.zeros((FINAL_BLOCK_C,), dtype=ct.float32)
    acc_x = ct.zeros((FINAL_BLOCK_C,), dtype=ct.float32)
    acc_masked = ct.zeros((FINAL_BLOCK_C,), dtype=ct.float32)

    n_row_tiles: ct.Constant[int] = M_C // FINAL_BLOCK_TILES
    for r in range(n_row_tiles):
        x_rhs_tile = ct.load(
            partial_x_rhs_ptr,
            index=(r, col_tile),
            shape=(FINAL_BLOCK_TILES, FINAL_BLOCK_C),
        )
        x_tile = ct.load(
            partial_x_ptr,
            index=(r, col_tile),
            shape=(FINAL_BLOCK_TILES, FINAL_BLOCK_C),
        )
        masked_tile = ct.load(
            partial_masked_ptr,
            index=(r, col_tile),
            shape=(FINAL_BLOCK_TILES, FINAL_BLOCK_C),
        )
        acc_x_rhs = acc_x_rhs + ct.sum(x_rhs_tile, axis=0)
        acc_x = acc_x + ct.sum(x_tile, axis=0)
        acc_masked = acc_masked + ct.sum(masked_tile, axis=0)

    ct.store(out_x_rhs_ptr, index=(col_tile,), tile=acc_x_rhs)
    ct.store(out_x_ptr, index=(col_tile,), tile=acc_x)
    ct.store(out_masked_ptr, index=(col_tile,), tile=acc_masked)


@oracle_impl(hardware="B200", point="0d0b1d4e", ROW_GROUP=22, XBLOCK=1, BLOCK_C=1024, FINAL_BLOCK_C=16, FINAL_BLOCK_TILES=512)
def oracle_forward(
    inputs,
    *,
    ROW_GROUP: int,
    XBLOCK: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    FINAL_BLOCK_TILES: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_shape_params = inputs
    device = arg0_1.device

    # arg2_1 is [32,512,768] but we work in the [M,C] view since M=32*512=16384.
    arg2_2d = arg2_1.view(M, C)
    arg3_1d = arg3_1.view(M)
    arg4_2d = arg4_1.view(M, C)

    padded_grad = torch.empty((M, BLOCK_C), device=device, dtype=torch.float32)
    padded_masked = torch.empty((M, BLOCK_C), device=device, dtype=torch.float32)
    padded_x_rhs = torch.empty((M, BLOCK_C), device=device, dtype=torch.float32)
    padded_x = torch.empty((M, BLOCK_C), device=device, dtype=torch.float32)
    padded_masked_col = torch.empty((M, BLOCK_C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (M, 1, 1),
        _row_ln_backward_kernel,
        (arg0_1, arg1_1, arg2_2d, arg3_1d, arg4_2d,
         padded_grad, padded_masked, padded_x_rhs, padded_x, padded_masked_col,
         C, BLOCK_C, DROP_SCALE),
    )

    # Slice back to [M, C] for the returned tensors.
    grad_out_2d = padded_grad[:, :C].contiguous()
    grad_out = grad_out_2d.view(32, 512, C)
    masked_out = padded_masked[:, :C].contiguous()

    # Column reductions performed by a second cuTile kernel over the padded
    # [M, BLOCK_C] partials (masked-out cols were zeroed in the row kernel).
    padded_out_x_rhs = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    padded_out_x = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    padded_out_masked = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)

    ct.launch(
        stream,
        (ct.cdiv(BLOCK_C, FINAL_BLOCK_C), 1, 1),
        _finalize_column_sums_kernel,
        (padded_x_rhs, padded_x, padded_masked_col,
         padded_out_x_rhs, padded_out_x, padded_out_masked,
         M, C, BLOCK_C, FINAL_BLOCK_C, FINAL_BLOCK_TILES),
    )

    out_x_rhs = padded_out_x_rhs[:C].contiguous()
    out_x = padded_out_x[:C].contiguous()
    out_masked = padded_out_masked[:C].contiguous()

    return (
        grad_out,
        out_x_rhs,
        out_x,
        masked_out,
        masked_out.permute(1, 0),
        out_masked,
    )
