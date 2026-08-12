"""cuTile port of var_mean_28b0ef07dcb9: GPT-J mixed-dtype residual
LayerNorm.

Row kernel: add lhs+rhs (bf16) then + residual (f32), compute variance
and mean, normalize with rsqrt(var+eps), apply affine, output normalized
(f32), final bf16, and invstd/HIDDEN.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _gptj_layernorm_kernel(
    lhs_ptr,          # bf16 (rows, HIDDEN)
    rhs_ptr,          # bf16 (rows, HIDDEN)
    residual_ptr,     # f32 (rows, HIDDEN)
    weight_ptr,       # f32 (HIDDEN,)
    bias_ptr,         # f32 (HIDDEN,)
    normalized_ptr,   # f32 (rows, HIDDEN)
    final_ptr,        # bf16 (rows, HIDDEN)
    invstd_div_ptr,   # f32 (rows,)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    lhs = ct.load(lhs_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    # Match eager: bf16 add lhs+rhs, then bf16+f32 -> f32.
    pair_bf16 = ct.astype(ct.astype(lhs, ct.float32) + ct.astype(rhs, ct.float32),
                          ct.bfloat16)
    pair_f = ct.astype(pair_bf16, ct.float32)
    x = pair_f + residual

    total = ct.sum(x)
    mean = total * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    final = ct.astype(affine, ct.bfloat16)

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(final_ptr, index=(row, 0), tile=final)
    invstd_div = invstd * (1.0 / HIDDEN)
    ct.store(invstd_div_ptr, index=(row,), tile=ct.reshape(invstd_div, (1,)))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="3fdaed2a", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    normalized_shape = _as_shape(shape0)
    final_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    stat_shape = normalized_shape[:-1] + (1,)

    normalized = torch.empty_strided(
        normalized_shape,
        _contiguous_stride(normalized_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    final_bf16 = torch.empty_strided(
        final_shape,
        _contiguous_stride(final_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    # arg2_1 is f32[1, 128, 4096] view as [128, 4096]
    residual_2d = arg2_1.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    invstd_1d = invstd_div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _gptj_layernorm_kernel,
        (arg0_1, arg1_1, residual_2d, arg3_1, arg4_1,
         normalized_2d, final_bf16, invstd_1d, hidden, BLOCK_H),
    )
    return normalized, final_bf16, invstd_div
