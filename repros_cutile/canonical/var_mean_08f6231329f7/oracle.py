"""cuTile port of var_mean_08f6231329f7: ALBERT tanh-GELU + LayerNorm.

Per row: tanh-approx GELU with bf16 rounding at each step, then
LayerNorm(hidden=128, eps=1e-12) with bf16 affine, output bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


def _bf16_round(x):
    return ct.astype(ct.astype(x, ct.bfloat16), ct.float32)


@ct.kernel
def _gelu_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf16 = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x = ct.astype(x_bf16, ct.float32)

    half = _bf16_round(x * 0.5)
    square = _bf16_round(x * x)
    cubic = _bf16_round(square * x)
    cubic_scaled = _bf16_round(cubic * 0.044715)
    poly = _bf16_round(x + cubic_scaled)
    tanh_arg = _bf16_round(poly * 0.7978845608028654)
    # cuTile has no tanh — express as (exp(2x) - 1) / (exp(2x) + 1)
    e2 = ct.exp(tanh_arg * 2.0)
    tanh_val_raw = (e2 - 1.0) / (e2 + 1.0)
    tanh_val = _bf16_round(tanh_val_raw)
    one_plus_tanh = _bf16_round(tanh_val + 1.0)
    gelu = _bf16_round(half * one_plus_tanh)

    inv_h = 1.0 / HIDDEN
    mean = ct.sum(gelu) * inv_h
    centered = gelu - mean
    variance = ct.sum(centered * centered) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="699e8097", BLOCK_H=128, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    x, weight, bias, _view_shape, out_shape = inputs
    output_shape = _shape_tuple(out_shape)
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    out = torch.empty_strided(
        output_shape, _contiguous_stride(output_shape),
        device=x.device, dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _gelu_layernorm_kernel,
        (x, weight, bias, out, hidden, BLOCK_H),
    )
    return out
