"""cuTile port of var_mean_a2329fda80e5: M2M100 residual-add LayerNorm with alias returns.

Per row: bf16 flat + bf16 residual rounded-through-bf16 → f32 x; pop var_mean;
rsqrt(var+1e-5); affine; bf16 output; return 24 aliased view returns.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _residual_layernorm_kernel(
    flat_ptr,      # bf16 (rows, HIDDEN)
    residual_ptr,  # bf16 (rows, HIDDEN)
    weight_ptr,    # bf16 (HIDDEN,)
    bias_ptr,      # bf16 (HIDDEN,)
    out_ptr,       # bf16 (rows, HIDDEN)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    x_bf16 = ct.astype(resid_f + flat_f, ct.bfloat16)
    x = ct.astype(x_bf16, ct.float32)

    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    out = ct.astype(centered * invstd * weight_2d + bias_2d, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="17affd46", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, *view_shapes = inputs
    base_shape = _as_shape(shape0)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])

    base = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    residual_2d = arg1_1.view(rows, hidden)
    base_2d = base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, base_2d, hidden, BLOCK_H),
    )
    return (base,) + tuple(base.view(_as_shape(shape)) for shape in view_shapes)
