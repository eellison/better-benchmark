"""cuTile port of var_mean_3c7707c0bb8a: ALBERT tanh-GELU + LayerNorm (bf16).

Uses ct.tanh directly. HIDDEN=128 for the point 'a0e6bc91' fits BLOCK_H=128.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _gelu_layernorm_kernel(
    x_ptr,       # bf16 [ROWS, HIDDEN]
    weight_ptr,  # f32 [HIDDEN]
    bias_ptr,    # f32 [HIDDEN]
    mean_ptr,    # f32 [ROWS]
    rsqrt_ptr,   # f32 [ROWS]
    out_ptr,     # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf16 = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x = ct.astype(x_bf16, ct.float32)

    half = ct.astype(ct.astype(x * 0.5, ct.bfloat16), ct.float32)
    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    gelu = half * (ct.tanh(tanh_arg) + 1.0)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    valid = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    reduce_input = ct.where(valid, gelu, 0.0)
    mean = ct.sum(reduce_input) * (1.0 / HIDDEN)
    centered = ct.where(valid, gelu - mean, 0.0)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-12)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.reshape(weight, (1, BLOCK_H))
    bias_f = ct.reshape(bias, (1, BLOCK_H))
    out = normalized * weight_f + bias_f

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="a0e6bc91", BLOCK_H=128)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    stat_shape = (view_shape[0], view_shape[1], 1)
    stat_stride = (view_shape[1], 1, 1)

    mean = torch.empty_strided(stat_shape, stat_stride, device=arg0_1.device, dtype=torch.float32)
    rsqrt = torch.empty_strided(stat_shape, stat_stride, device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape), device=arg0_1.device, dtype=torch.bfloat16
    )

    mean_flat = mean.view(rows)
    rsqrt_flat = rsqrt.view(rows)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _gelu_layernorm_kernel,
        (arg0_1, arg1_1, arg2_1, mean_flat, rsqrt_flat, out, hidden, BLOCK_H),
    )
    return mean, rsqrt, out
