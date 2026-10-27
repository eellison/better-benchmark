"""cuTile port of var_mean_d552da4c6138: SigLIP hidden-768 LayerNorm.

Row-wise LayerNorm over HIDDEN=768. HIDDEN is not a power of 2, so we load
with BLOCK_H=1024 and column-mask the reductions, then store to a padded
buffer and slice back to [ROWS, HIDDEN].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _layernorm_kernel(
    x_ptr,          # bf16 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    mean_ptr,       # f32 [ROWS]
    rsqrt_ptr,      # f32 [ROWS]
    out_ptr,        # bf16 [ROWS, BLOCK_H]  (padded, sliced by caller)
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN_C
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))
    zero = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)

    inv_h = 1.0 / HIDDEN_C
    x_masked = ct.where(col_mask_2d, x_f, zero)
    mean = ct.sum(x_masked) * inv_h
    centered = x_f - mean
    centered_masked = ct.where(col_mask_2d, centered, zero)
    variance = ct.sum(centered_masked * centered_masked) * inv_h
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))

    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d
    out = ct.astype(affine, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)

    # Store mean and rsqrt as size-(1,) tiles.
    mean_tile = ct.full((1,), 0.0, dtype=ct.float32)
    invstd_tile = ct.full((1,), 0.0, dtype=ct.float32)
    # We need the scalars as a length-1 tile — reuse using reshape.
    mean_scalar = ct.reshape(mean_tile, (1,)) * 0.0 + mean
    invstd_scalar = ct.reshape(invstd_tile, (1,)) * 0.0 + invstd
    ct.store(mean_ptr, index=(row,), tile=mean_scalar)
    ct.store(rsqrt_ptr, index=(row,), tile=invstd_scalar)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="0e852c6f", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    rows = int(out_shape[0])
    hidden = int(out_shape[1])

    mean = torch.empty_strided(
        (view_shape[0], view_shape[1], 1),
        (1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (view_shape[0], view_shape[1], 1),
        (1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    # Padded output tensor to accommodate BLOCK_H > HIDDEN.
    padded = torch.empty(
        (rows, BLOCK_H),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # Flatten mean/rsqrt to 1D views for the kernel.
    mean_flat = mean.view(rows)
    rsqrt_flat = rsqrt.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layernorm_kernel,
        (arg0_1, arg1_1, arg2_1, mean_flat, rsqrt_flat, padded, hidden, BLOCK_H),
    )

    out = padded[:, :hidden].contiguous()
    return arg0_1.view(view_shape), mean, rsqrt, out
