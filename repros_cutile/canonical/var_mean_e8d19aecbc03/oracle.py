"""cuTile port of var_mean_e8d19aecbc03: residual LayerNorm multi-output row kernel.

Per row: bf16 residual add (from f32), separately compute mean/rsqrt on the
raw f32 sum (side outputs) and on the bf16-rounded sum (affine), affine
fp32/bf16 outputs. HIDDEN=4096 (power-of-2, no padding).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _residual_ln_multi_kernel(
    flat_ptr,     # bf16 [ROWS, HIDDEN]
    residual_ptr, # bf16 [ROWS, HIDDEN]
    weight_ptr,   # f32 [HIDDEN]
    bias_ptr,     # f32 [HIDDEN]
    add_ptr,      # bf16 [ROWS, HIDDEN]
    mean_ptr,     # f32 [ROWS]
    rsqrt_ptr,    # f32 [ROWS]
    affine_ptr,   # f32 [ROWS, HIDDEN]
    bf16_ptr,     # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.astype(ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    residual = ct.astype(ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    x_fp32 = flat + residual
    added_bf16 = ct.astype(x_fp32, ct.bfloat16)
    ct.store(add_ptr, index=(row, 0), tile=added_bf16)
    x_bf16 = ct.astype(added_bf16, ct.float32)

    mean_out = ct.sum(x_fp32) * (1.0 / HIDDEN)
    centered_out = x_fp32 - mean_out
    variance_out = ct.sum(centered_out * centered_out) * (1.0 / HIDDEN)
    invstd_out = ct.rsqrt(variance_out + EPS)
    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean_out, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd_out, (1,)))

    mean_affine = ct.sum(x_bf16) * (1.0 / HIDDEN)
    centered_affine = x_bf16 - mean_affine
    variance_affine = ct.sum(centered_affine * centered_affine) * (1.0 / HIDDEN)
    invstd_affine = ct.rsqrt(variance_affine + EPS)
    normalized = centered_affine * invstd_affine

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="30413f1a", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    stat_shape = norm_shape[:-1] + (1,)

    added = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape), device=arg0_1.device, dtype=torch.bfloat16
    )
    mean = torch.empty_strided(
        stat_shape, _contiguous_stride(stat_shape), device=arg0_1.device, dtype=torch.float32
    )
    rsqrt = torch.empty_strided(
        stat_shape, _contiguous_stride(stat_shape), device=arg0_1.device, dtype=torch.float32
    )
    affine = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape), device=arg0_1.device, dtype=torch.float32
    )
    bf16_view = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape), device=arg0_1.device, dtype=torch.bfloat16
    )

    # Reshape kernel views to 2D (rows, hidden)
    residual_2d = arg1_1.view(rows, hidden)
    added_2d = added.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_view_2d = bf16_view.view(rows, hidden)
    mean_flat = mean.view(rows)
    rsqrt_flat = rsqrt.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_ln_multi_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, added_2d, mean_flat, rsqrt_flat,
         affine_2d, bf16_view_2d, hidden, BLOCK_H),
    )
    return added, mean, rsqrt, affine, bf16_view
