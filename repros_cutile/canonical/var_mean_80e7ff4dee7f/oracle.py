"""cuTile port of var_mean_80e7ff4dee7f: Longformer inference residual
LayerNorm. bf16 residual add -> f32 var/mean over hidden -> rsqrt-affine
-> bf16 output.

The Triton oracle uses an "anchor + tolerance" clamp against a bf16-rounded
midpoint to preserve Inductor bit-exactness. For the numerics check at
atol/rtol=1e-2, a straight f32 LN suffices.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _residual_layernorm_kernel(
    flat_ptr,       # bf16 (rows, HIDDEN)
    residual_ptr,   # bf16 (rows, HIDDEN)
    weight_ptr,     # bf16 (HIDDEN,)
    bias_ptr,       # bf16 (HIDDEN,)
    out_ptr,        # bf16 (rows, HIDDEN)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(
        flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    added_f = flat_f + resid_f
    added_bf16 = ct.astype(added_f, ct.bfloat16)
    x = ct.astype(added_bf16, ct.float32)
    x = ct.where(col_mask_2d, x, 0.0)

    total = ct.sum(x)
    mean = total * (1.0 / HIDDEN)
    centered = ct.where(col_mask_2d, x - mean, 0.0)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="1cea4d76", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_shape = _as_shape(_shape_param_0)

    residual_2d = arg1_1.view(rows, hidden)

    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    block_h = 1 << (hidden - 1).bit_length()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, out_2d, hidden, block_h),
    )
    return out
