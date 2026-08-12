"""cuTile port of sum_sum_sum_31ea6dc807dd: OPT bf16 LayerNorm/dropout backward
cooperative multi-output split-K.

Two sequential kernels:
1. `_row_tile_kernel` — one program per tile of ROWS_PER_TILE rows. Loads the
   row inputs (x, residual, mask, bias, mean, invstd, grad_in), recomputes the
   normalized residual branch, writes the f32 `[4, 2048, 768]` gradient side
   output AND the bf16 `[8192, 768]` masked output, and accumulates per-tile
   `[3, HIDDEN]` column partials for (x*norm, x, masked).
2. `_finalize_kernel` — sums partials across tiles into three `[HIDDEN]` sums.
   The masked-sum output is rounded through bf16 as the Triton oracle does.

HIDDEN=768 is not a power of two, so tile shape uses BLOCK_D=1024 with a
col-mask; reductions treat OOB columns as zero.

Triton's inline_asm `add.rn.f32`/`sub.rn.f32`/`mul.rn.f32` all map to cuTile's
default round-to-nearest-even fp32 arithmetic. `to(bfloat16,
fp_downcast_rounding='rtne')` becomes `ct.astype(_, ct.bfloat16)`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 8192
D = 768
DROP_SCALE = 1.1111111111111112
INV_D = 1.0 / 768.0


@ct.kernel
def _row_tile_kernel(
    x_ptr,           # bf16 [M, D]
    gamma_ptr,       # f32  [D]
    residual_ptr,    # bf16 [M, D]
    mask_ptr,        # bool [M, D]
    bias_ptr,        # f32  [M, D]
    mean_ptr,        # f32  [M]
    invstd_ptr,      # f32  [M]
    grad_in_ptr,     # f32  [M, D]
    grad_out_ptr,    # f32  [M, D]
    masked_out_ptr,  # bf16 [M, D]
    partials_ptr,    # f32  [NUM_TILES, 3, D]
    NUM_TILES: ct.Constant[int],
    D_: ct.Constant[int],
    ROWS_PER_TILE: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    tile = ct.bid(0)
    cols = ct.arange(BLOCK_D, dtype=ct.int32)
    col_valid = cols < D_

    gamma_row = ct.load(gamma_ptr, index=(0,), shape=(BLOCK_D,),
                        padding_mode=ct.PaddingMode.ZERO)
    gamma = ct.astype(gamma_row, ct.float32)
    zero_1d = ct.zeros((BLOCK_D,), dtype=ct.float32)
    gamma = ct.where(col_valid, gamma, zero_1d)

    acc_x_norm = ct.zeros((BLOCK_D,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_D,), dtype=ct.float32)
    acc_masked = ct.zeros((BLOCK_D,), dtype=ct.float32)

    for row_off in range(ROWS_PER_TILE):
        row = tile * ROWS_PER_TILE + row_off

        x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_D),
                       padding_mode=ct.PaddingMode.ZERO)
        residual_bf = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_D),
                              padding_mode=ct.PaddingMode.ZERO)
        mask_b = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_D),
                         padding_mode=ct.PaddingMode.ZERO)
        bias_f = ct.load(bias_ptr, index=(row, 0), shape=(1, BLOCK_D),
                         padding_mode=ct.PaddingMode.ZERO)
        mean_v = ct.load(mean_ptr, index=(row,), shape=(1,))
        invstd_v = ct.load(invstd_ptr, index=(row,), shape=(1,))
        grad_in_f = ct.load(grad_in_ptr, index=(row, 0), shape=(1, BLOCK_D),
                            padding_mode=ct.PaddingMode.ZERO)

        # Squeeze the leading axis: work with (BLOCK_D,) tiles.
        x_bf_1d = ct.reshape(x_bf, (BLOCK_D,))
        residual_bf_1d = ct.reshape(residual_bf, (BLOCK_D,))
        mask_1d = ct.reshape(mask_b, (BLOCK_D,))
        bias_1d = ct.reshape(bias_f, (BLOCK_D,))
        grad_in_1d = ct.reshape(grad_in_f, (BLOCK_D,))
        mean_s = ct.reshape(mean_v, ())
        invstd_s = ct.reshape(invstd_v, ())

        x = ct.astype(x_bf_1d, ct.float32)
        residual = ct.astype(residual_bf_1d, ct.float32)

        # Residual branch: mask * residual (bf16), scale by 1.111 (bf16 boundary).
        residual_masked_bf = ct.astype(
            ct.astype(mask_1d, ct.float32) * residual, ct.bfloat16
        )
        residual_scaled_bf = ct.astype(
            ct.astype(residual_masked_bf, ct.float32) * DROP_SCALE, ct.bfloat16
        )
        norm = (bias_1d + ct.astype(residual_scaled_bf, ct.float32) - mean_s) * invstd_s

        # Reductions with OOB columns zeroed.
        weighted = x * gamma
        weighted_masked = ct.where(col_valid, weighted, zero_1d)
        row_sum_s = ct.reshape(ct.sum(weighted_masked), ())
        weighted_norm = weighted * norm
        row_dot_s = ct.reshape(ct.sum(ct.where(col_valid, weighted_norm, zero_1d)), ())

        grad = grad_in_1d + (invstd_s * INV_D) * (
            weighted * float(D_) - row_sum_s - norm * row_dot_s
        )
        # grad is stored ONLY in valid cols; use where to zero OOB, then store.
        grad_valid = ct.where(col_valid, grad, zero_1d)
        ct.store(grad_out_ptr, index=(row, 0), tile=ct.reshape(grad_valid, (1, BLOCK_D)))

        grad_bf = ct.astype(grad, ct.bfloat16)
        keep_scaled_bf = ct.astype(
            ct.astype(mask_1d, ct.float32) * DROP_SCALE, ct.bfloat16
        )
        masked_bf = ct.astype(
            ct.astype(grad_bf, ct.float32) * ct.astype(keep_scaled_bf, ct.float32),
            ct.bfloat16,
        )
        # Zero OOB for the bf16 store.
        zero_bf = ct.full((BLOCK_D,), 0.0, dtype=ct.bfloat16)
        masked_bf_valid = ct.where(col_valid, masked_bf, zero_bf)
        ct.store(masked_out_ptr, index=(row, 0),
                 tile=ct.reshape(masked_bf_valid, (1, BLOCK_D)))

        # Accumulate column partials.
        acc_x_norm = acc_x_norm + ct.where(col_valid, x * norm, zero_1d)
        acc_x = acc_x + ct.where(col_valid, x, zero_1d)
        acc_masked = acc_masked + ct.where(col_valid,
                                            ct.astype(masked_bf, ct.float32),
                                            zero_1d)

    # Write partials at [tile, 0..2, :].
    ct.store(partials_ptr, index=(tile, 0, 0),
             tile=ct.reshape(acc_x_norm, (1, 1, BLOCK_D)))
    ct.store(partials_ptr, index=(tile, 1, 0),
             tile=ct.reshape(acc_x, (1, 1, BLOCK_D)))
    ct.store(partials_ptr, index=(tile, 2, 0),
             tile=ct.reshape(acc_masked, (1, 1, BLOCK_D)))


@ct.kernel
def _finalize_kernel(
    partials_ptr,        # f32 [NUM_TILES, 3, D_STORE]
    out_x_norm_ptr,      # f32 [D]
    out_x_ptr,           # f32 [D]
    out_masked_sum_ptr,  # f32 [D]
    NUM_TILES: ct.Constant[int],
    D_: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    col_block = ct.bid(0)
    cols = ct.arange(BLOCK_D, dtype=ct.int32) + col_block * BLOCK_D
    col_valid = cols < D_

    acc_xn = ct.zeros((BLOCK_D,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_D,), dtype=ct.float32)
    acc_m = ct.zeros((BLOCK_D,), dtype=ct.float32)

    num_iters = (NUM_TILES + BLOCK_TILES - 1) // BLOCK_TILES
    for i in range(num_iters):
        tiles = ct.arange(BLOCK_TILES, dtype=ct.int32) + i * BLOCK_TILES
        tile_valid = tiles < NUM_TILES
        tile_valid_2d = ct.reshape(tile_valid, (BLOCK_TILES, 1))
        col_valid_2d = ct.reshape(col_valid, (1, BLOCK_D))
        mask2 = ct.broadcast_to(tile_valid_2d, (BLOCK_TILES, BLOCK_D)) & ct.broadcast_to(
            col_valid_2d, (BLOCK_TILES, BLOCK_D)
        )
        tiles_2d = ct.reshape(tiles, (BLOCK_TILES, 1))
        cols_2d = ct.reshape(cols, (1, BLOCK_D))
        tiles_broad = ct.broadcast_to(tiles_2d, (BLOCK_TILES, BLOCK_D))
        cols_broad = ct.broadcast_to(cols_2d, (BLOCK_TILES, BLOCK_D))
        zero_i32_2d = ct.zeros((BLOCK_TILES, BLOCK_D), dtype=ct.int32)
        safe_tiles = ct.where(mask2, tiles_broad, zero_i32_2d)
        safe_cols = ct.where(mask2, cols_broad, zero_i32_2d)
        zero_idx = ct.zeros((BLOCK_TILES, BLOCK_D), dtype=ct.int32)

        xn = ct.gather(partials_ptr, (safe_tiles, zero_idx, safe_cols),
                       mask=mask2, padding_value=0.0)
        x = ct.gather(partials_ptr,
                      (safe_tiles, ct.ones((BLOCK_TILES, BLOCK_D), dtype=ct.int32),
                       safe_cols),
                      mask=mask2, padding_value=0.0)
        m = ct.gather(partials_ptr,
                      (safe_tiles,
                       ct.full((BLOCK_TILES, BLOCK_D), 2, dtype=ct.int32),
                       safe_cols),
                      mask=mask2, padding_value=0.0)

        zero2 = ct.zeros((BLOCK_TILES, BLOCK_D), dtype=ct.float32)
        acc_xn = acc_xn + ct.sum(ct.where(mask2, xn, zero2), axis=0)
        acc_x = acc_x + ct.sum(ct.where(mask2, x, zero2), axis=0)
        acc_m = acc_m + ct.sum(ct.where(mask2, m, zero2), axis=0)

    zero_1d = ct.zeros((BLOCK_D,), dtype=ct.float32)
    ct.scatter(out_x_norm_ptr, (cols,), ct.where(col_valid, acc_xn, zero_1d),
               mask=col_valid)
    ct.scatter(out_x_ptr, (cols,), ct.where(col_valid, acc_x, zero_1d),
               mask=col_valid)
    # bf16 roundtrip on masked sum
    acc_m_bf = ct.astype(acc_m, ct.bfloat16)
    acc_m_f = ct.astype(acc_m_bf, ct.float32)
    ct.scatter(out_masked_sum_ptr, (cols,),
               ct.where(col_valid, acc_m_f, zero_1d), mask=col_valid)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="81e4a7b8",
    ROWS_PER_TILE=16,
    BLOCK_D=1024,
    FINAL_BLOCK_TILES=64,
    FINAL_BLOCK_D=16,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_TILE: int,
    BLOCK_D: int,
    FINAL_BLOCK_TILES: int,
    FINAL_BLOCK_D: int,
):
    (
        x_bf16,      # bf16 [M, D]
        gamma,       # f32  [D]
        residual,    # bf16 [M, D]
        mask,        # bool [4, 2048, D]
        bias,        # f32  [4, 2048, D]
        mean,        # f32  [M, 1]
        invstd,      # f32  [M, 1]
        grad_in,     # f32  [M, D]
        _shape0,
        _shape1,
        _shape2,
        _shape3,
        _shape4,
    ) = inputs
    del _shape0, _shape1, _shape2, _shape3, _shape4

    device = x_bf16.device
    # Views. mask/bias come in as [4, 2048, D]; reshape to [M, D].
    mask_2d = mask.reshape(M, D)
    bias_2d = bias.reshape(M, D)
    mean_1d = mean.view(M)
    invstd_1d = invstd.view(M)

    grad_out_2d = torch.empty((M, D), device=device, dtype=torch.float32)
    masked_out_2d = torch.empty((M, D), device=device, dtype=torch.bfloat16)

    num_tiles = M // ROWS_PER_TILE
    partials = torch.empty((num_tiles, 3, D), device=device, dtype=torch.float32)

    out_x_norm = torch.empty((D,), device=device, dtype=torch.float32)
    out_x = torch.empty((D,), device=device, dtype=torch.float32)
    out_masked = torch.empty((D,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_tiles, 1, 1),
        _row_tile_kernel,
        (
            x_bf16,
            gamma,
            residual,
            mask_2d,
            bias_2d,
            mean_1d,
            invstd_1d,
            grad_in,
            grad_out_2d,
            masked_out_2d,
            partials,
            num_tiles,
            D,
            ROWS_PER_TILE,
            BLOCK_D,
        ),
    )
    ct.launch(
        stream,
        ((D + FINAL_BLOCK_D - 1) // FINAL_BLOCK_D, 1, 1),
        _finalize_kernel,
        (
            partials,
            out_x_norm,
            out_x,
            out_masked,
            num_tiles,
            D,
            FINAL_BLOCK_TILES,
            FINAL_BLOCK_D,
        ),
    )

    # Reshape to Repro's expected shapes.
    grad_out = torch.empty_strided(
        (4, 2048, D), (2048 * D, D, 1), device=device, dtype=torch.float32,
    )
    grad_out.copy_(grad_out_2d.view(4, 2048, D))

    return (
        out_x_norm,          # sum_3
        out_x,               # sum_4
        grad_out,            # view_2
        masked_out_2d,       # view_3
        masked_out_2d.permute(1, 0),  # permute
        out_masked,          # sum_5_f (bf16-roundtripped)
    )
