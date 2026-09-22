"""cuTile port of var_mean_f7a754994b8a: DINOv2 scaled-residual LayerNorm.

Ports the Triton `_scaled_residual_layernorm_bf16_kernel` — inline PTX RN
rounding is just cuTile's default IEEE rtne arithmetic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@ct.kernel
def _scaled_residual_layernorm_kernel(
    source_ptr,      # bf16 [ROWS, HIDDEN]
    scale_ptr,       # bf16 [HIDDEN]
    residual_ptr,    # bf16 [ROWS, HIDDEN]
    weight_ptr,      # bf16 [HIDDEN]
    bias_ptr,        # bf16 [HIDDEN]
    add_out_ptr,     # bf16 [ROWS, HIDDEN]
    norm_out_ptr,    # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    HIDDEN_PAD: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)

    source = ct.load(source_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                     padding_mode=ct.PaddingMode.ZERO)
    scale = ct.load(scale_ptr, index=(0,), shape=(HIDDEN_PAD,),
                    padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                       padding_mode=ct.PaddingMode.ZERO)

    source_f = ct.astype(source, ct.float32)
    scale_f = ct.astype(scale, ct.float32)
    residual_f = ct.astype(residual, ct.float32)

    scale_2d = ct.reshape(scale_f, (1, HIDDEN_PAD))
    scaled_bf16 = ct.astype(source_f * scale_2d, ct.bfloat16)
    scaled = ct.astype(scaled_bf16, ct.float32)
    x_bf16 = ct.astype(residual_f + scaled, ct.bfloat16)
    x = ct.astype(x_bf16, ct.float32)

    ct.store(add_out_ptr, index=(row, 0), tile=x_bf16)

    cols = ct.arange(HIDDEN_PAD, dtype=ct.int32)
    valid = cols < HIDDEN
    valid_2d = ct.reshape(valid, (1, HIDDEN_PAD))
    zero_f = ct.full(shape=(1, HIDDEN_PAD), fill_value=0.0, dtype=ct.float32)

    x_for_reduce = ct.where(valid_2d, x, zero_f)
    mean = ct.sum(x_for_reduce) / HIDDEN_F
    centered = x - mean
    centered_masked = ct.where(valid_2d, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) / HIDDEN_F
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, HIDDEN_PAD))
    bias_2d = ct.reshape(bias_f, (1, HIDDEN_PAD))

    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=affine_bf16)


@oracle_impl(hardware="B200", point="01a8f9c7", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    source, scale, residual, weight, bias, add_shape, norm_shape = inputs
    add_shape = tuple(int(dim) for dim in add_shape)
    norm_shape = tuple(int(dim) for dim in norm_shape)
    hidden = int(source.shape[1])
    rows = int(source.shape[0])
    HIDDEN_PAD = _next_pow2(hidden)

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=source.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=source.device,
        dtype=torch.bfloat16,
    )
    residual_2d = residual.view(rows, hidden)
    add_out_2d = add_out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _scaled_residual_layernorm_kernel,
        (source, scale, residual_2d, weight, bias, add_out_2d, norm_out,
         hidden, HIDDEN_PAD, float(hidden)),
    )
    return add_out, norm_out
