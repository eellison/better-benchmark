"""cuTile port of var_mean_cf9088d163f5: DINOv2 residual LayerNorm + selected-token affine.

Uses padding_mode=ZERO on tile loads/stores to avoid materializing padded
scratch buffers. HIDDEN=768 (non-pow2) is padded up to BLOCK_H=1024 in-tile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 175360
BATCH = 128
TOKENS = 1370
HIDDEN = 768
BLOCK_H = 1024
EPS = 1.0e-6

ACT_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
SELECTED_SHAPE = (BATCH, HIDDEN)
SELECTED_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, TOKENS, 1)
SIDE_STRIDE = (TOKENS, 1, 1)


@ct.kernel
def _dinov2_kernel(
    input_bf16_ptr,   # bf16 [ROWS, HIDDEN]
    gamma_ptr,        # f32 [HIDDEN]
    residual_ptr,     # f32 [ROWS, HIDDEN]  (viewed as flat)
    weight_ptr,       # f32 [HIDDEN]
    bias_ptr,         # f32 [HIDDEN]
    norm_out_ptr,     # f32 [ROWS, HIDDEN]
    selected_out_ptr, # f32 [BATCH, HIDDEN]
    side_out_ptr,     # f32 [ROWS]
    HIDDEN_: ct.Constant[int],
    TOKENS_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    # Load using padding_mode=ZERO to handle HIDDEN < BLOCK_H_ without a copy.
    input_bf16 = ct.astype(
        ct.load(input_bf16_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    gamma = ct.reshape(
        ct.load(gamma_ptr, index=(0,), shape=(BLOCK_H_,),
                padding_mode=ct.PaddingMode.ZERO),
        (1, BLOCK_H_),
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    scaled = input_bf16 * gamma
    x = residual + scaled

    # Mask OOB (HIDDEN..BLOCK_H-1) to zero for reductions.
    c_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    valid = ct.reshape(c_idx < HIDDEN_, (1, BLOCK_H_))
    zero_2d = ct.zeros((1, BLOCK_H_), dtype=ct.float32)
    x_masked = ct.where(valid, x, zero_2d)

    mean = ct.sum(x_masked) * (1.0 / HIDDEN_)
    centered = ct.where(valid, x - mean, zero_2d)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd
    # Store direct — OOB columns are zero anyway (from ct.where) and the
    # output row is exactly HIDDEN wide; cuTile drops OOB writes.
    ct.store(norm_out_ptr, index=(row, 0), tile=normalized)
    ct.store(side_out_ptr, index=(row,),
             tile=ct.reshape(invstd * (1.0 / HIDDEN_), (1,)))

    # Affine only for token 0.
    weight = ct.reshape(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,),
                padding_mode=ct.PaddingMode.ZERO),
        (1, BLOCK_H_),
    )
    bias = ct.reshape(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_,),
                padding_mode=ct.PaddingMode.ZERO),
        (1, BLOCK_H_),
    )
    affine = normalized * weight + bias
    is_token0 = row - (row // TOKENS_) * TOKENS_ == 0
    if is_token0:
        batch = row // TOKENS_
        ct.store(selected_out_ptr, index=(batch, 0), tile=affine)


@oracle_impl(hardware="B200", point="70d2be15", BLOCK_H=BLOCK_H)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0 = inputs
    norm_shape = tuple(int(dim) for dim in shape0)
    device = arg0_1.device

    # Output tensors: use natural shapes/strides — no padded scratch buffers.
    norm_out = torch.empty_strided(
        norm_shape, ACT_STRIDE, device=device, dtype=torch.float32
    )
    selected_out = torch.empty_strided(
        SELECTED_SHAPE, SELECTED_STRIDE, device=device, dtype=torch.float32
    )
    side_out = torch.empty_strided(
        SIDE_SHAPE, SIDE_STRIDE, device=device, dtype=torch.float32
    )

    # 2D views for the kernel (kernel indexes as (row, col)).
    input_2d = arg0_1.view(ROWS, HIDDEN)
    residual_2d = arg2_1.view(ROWS, HIDDEN)
    norm_2d = norm_out.view(ROWS, HIDDEN)
    side_flat = side_out.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _dinov2_kernel,
        (input_2d, arg1_1, residual_2d, arg3_1, arg4_1,
         norm_2d, selected_out, side_flat,
         HIDDEN, TOKENS, BLOCK_H),
    )

    return norm_out, selected_out, side_out
