"""cuTile port of var_mean_3ad14c3d031f: OPT residual-add LayerNorm.

For each row of [8192,768] bf16 x + bf16 residual: compute add (rounded to
bf16), then LayerNorm across HIDDEN=768 (non-power-of-2, so BLOCK_H=1024 with
zero-padded load + column-masked reductions). Output the rounded add tensor
plus the LayerNorm result and its three alias views.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
EPS = 1.0e-5


@ct.kernel
def _residual_layernorm_kernel(
    arg0_ptr,   # bf16 [ROWS, HIDDEN]
    arg1_ptr,   # bf16 [ROWS, HIDDEN]
    weight_ptr, # bf16 [HIDDEN]
    bias_ptr,   # bf16 [HIDDEN]
    add_out,    # bf16 [ROWS, BLOCK_H] (padded)
    norm_out,   # bf16 [ROWS, BLOCK_H] (padded)
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    lhs = ct.load(arg0_ptr, index=(row, 0), shape=(1, BLOCK_H),
                  padding_mode=ct.PaddingMode.ZERO)
    rhs = ct.load(arg1_ptr, index=(row, 0), shape=(1, BLOCK_H),
                  padding_mode=ct.PaddingMode.ZERO)

    add_f32 = ct.astype(lhs, ct.float32) + ct.astype(rhs, ct.float32)
    add_bf16 = ct.astype(add_f32, ct.bfloat16)
    ct.store(add_out, index=(row, 0), tile=add_bf16)

    x = ct.astype(add_bf16, ct.float32)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN_C
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))
    zero = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)

    inv_h = 1.0 / HIDDEN_C
    x_masked = ct.where(col_mask_2d, x, zero)
    mean = ct.sum(x_masked) * inv_h
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, zero)
    variance = ct.sum(centered_masked * centered_masked) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    out_bf16 = ct.astype(affine, ct.bfloat16)
    ct.store(norm_out, index=(row, 0), tile=out_bf16)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="e4faf4aa", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    add_shape = _shape_tuple(shape0)
    norm_shape = _shape_tuple(shape1)
    hidden = int(arg0_1.shape[1])

    padded_add = torch.empty(
        (ROWS, BLOCK_H),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    padded_norm = torch.empty(
        (ROWS, BLOCK_H),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _residual_layernorm_kernel,
        (arg0_1, arg1_1, arg2_1, arg3_1, padded_add, padded_norm, hidden, BLOCK_H),
    )

    add_out = padded_add[:, :hidden].contiguous().view(add_shape)
    norm_out = padded_norm[:, :hidden].contiguous()
    return (
        add_out,
        norm_out,
        norm_out.view(_shape_tuple(shape2)),
        norm_out.view(_shape_tuple(shape3)),
    )
