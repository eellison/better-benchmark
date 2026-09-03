"""cuTile port of sum_sum_7ce96d2e23ba: T5 RMSNorm/dropout backward.

Produces:
  - column weight-gradient vector [H] (f32)
  - full [8, 1024, H] add tensor (f32)
  - bf16 dropout-gradient tensor [8192, H]
  - transposed view of it [H, 8192]
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 8192
H = 512
SCALE_X = 0.04419417382415922
MASK_SCALE = 1.1111111111111112
INV_H = 1.0 / H


@ct.kernel
def _row_kernel(
    x_ptr,           # bf16 [M, H]
    mask0_ptr,       # b8 [M, H]
    weight_ptr,      # f32 [H]
    activ_ptr,       # f32 [M, H]
    row_scale_ptr,   # f32 [M]
    mask1_ptr,       # b8 [M, H]
    partial_ptr,     # f32 [num_row_blocks, H]
    add_out_ptr,     # f32 [M, H]
    bf16_out_ptr,    # bf16 [M, H]
    BLOCK_M: ct.Constant[int],
    H_: ct.Constant[int],
):
    row_block = ct.bid(0)
    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, H_))
    mask0 = ct.load(mask0_ptr, index=(row_block, 0), shape=(BLOCK_M, H_))
    activ = ct.load(activ_ptr, index=(row_block, 0), shape=(BLOCK_M, H_))
    mask1 = ct.load(mask1_ptr, index=(row_block, 0), shape=(BLOCK_M, H_))
    weight = ct.load(weight_ptr, index=(0,), shape=(H_,))
    row_scale = ct.load(row_scale_ptr, index=(row_block,), shape=(BLOCK_M,))

    x_f = ct.astype(x, ct.float32)
    mask0_f = ct.astype(mask0, ct.float32)
    weight_f = ct.reshape(weight, (1, H_))
    activ_f = activ
    mask1_f = ct.astype(mask1, ct.float32)
    row_scale_2d = ct.reshape(row_scale, (BLOCK_M, 1))

    scaled_x = x_f * SCALE_X
    scaled_mask0 = mask0_f * MASK_SCALE
    producer = scaled_x * scaled_mask0

    mul4 = activ_f * row_scale_2d
    col_term = producer * mul4
    # partial column sum for weight-gradient
    partial = ct.sum(col_term, axis=0)  # (H_,)
    partial_2d = ct.reshape(partial, (1, H_))
    ct.store(partial_ptr, index=(row_block, 0), tile=partial_2d)

    weighted = producer * weight_f
    row_term = weighted * activ_f
    row_sum = ct.sum(row_term, axis=1)  # (BLOCK_M,)

    first = weighted * row_scale_2d
    row_scale_sq = row_scale * row_scale
    row_scale_cu = row_scale_sq * row_scale
    norm_term = row_sum * -0.5 * row_scale_cu * INV_H
    norm_term_2d = ct.reshape(norm_term, (BLOCK_M, 1))
    activ_twice = activ_f * 2.0
    second = norm_term_2d * activ_twice
    add_out = first + second
    ct.store(add_out_ptr, index=(row_block, 0), tile=add_out)

    add_bf16 = ct.astype(ct.astype(add_out, ct.bfloat16), ct.float32)
    mask1_scaled = ct.astype(ct.astype(mask1_f * MASK_SCALE, ct.bfloat16), ct.float32)
    bf16_out = ct.astype(add_bf16 * mask1_scaled, ct.bfloat16)
    ct.store(bf16_out_ptr, index=(row_block, 0), tile=bf16_out)


@ct.kernel
def _finalize_col_sum_kernel(
    partial_ptr,   # f32 [num_row_blocks, H]
    out_ptr,       # f32 [H]
    NUM_BLOCKS: ct.Constant[int],
    H_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    tile = ct.load(partial_ptr, index=(0, col_block), shape=(BLOCK_R, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    total = ct.sum(tile, axis=0)
    ct.store(out_ptr, index=(col_block,), tile=total)


@oracle_impl(hardware="B200", point="a4350e46")
def oracle_forward(inputs):
    x, mask0, weight, activ, row_scale, mask1, _s3, _sv, _s3b, _s2 = inputs
    # Reshape everything to 2D [M, H]
    x_2d = x.view(M, H)
    mask0_2d = mask0.view(M, H)
    activ_2d = activ.view(M, H)
    mask1_2d = mask1.view(M, H)
    row_scale_1d = row_scale.view(M)

    BLOCK_M = 2
    num_row_blocks = M // BLOCK_M
    partial = torch.empty((num_row_blocks, H), device=x.device, dtype=torch.float32)
    add_out = torch.empty_strided((8, 1024, H), (1024 * H, H, 1),
                                   device=x.device, dtype=torch.float32)
    bf16_out = torch.empty_strided((M, H), (H, 1), device=x.device, dtype=torch.bfloat16)
    vector_out = torch.empty((H,), device=x.device, dtype=torch.float32)

    add_out_2d = add_out.view(M, H)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_blocks, 1, 1),
        _row_kernel,
        (x_2d, mask0_2d, weight, activ_2d, row_scale_1d, mask1_2d,
         partial, add_out_2d, bf16_out, BLOCK_M, H),
    )
    BLOCK_R = 1
    while BLOCK_R < num_row_blocks:
        BLOCK_R *= 2
    BLOCK_C = 32
    ct.launch(
        stream, (H // BLOCK_C, 1, 1),
        _finalize_col_sum_kernel,
        (partial, vector_out, num_row_blocks, H, BLOCK_R, BLOCK_C),
    )
    return vector_out, add_out, bf16_out, bf16_out.permute(1, 0)
