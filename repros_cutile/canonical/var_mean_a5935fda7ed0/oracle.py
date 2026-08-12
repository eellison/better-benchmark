"""cuTile port of var_mean_a5935fda7ed0: Albert bf16-residual LayerNorm.

Per row: promote flat bf16 + residual f32 to f32, compute mean/var/rsqrt
(eps=1e-12), emit normalized fp32 side tensor, affine to bf16 with weight
(f32) and bias (f32), plus rsqrt / hidden side output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 4096
EPS = 1.0e-12


@ct.kernel
def _add_layernorm_kernel(
    flat_bf16_ptr,   # bf16 [rows, HIDDEN]
    residual_ptr,    # f32  [rows, HIDDEN]
    weight_ptr,      # f32  [HIDDEN]
    bias_ptr,        # f32  [HIDDEN]
    normalized_ptr,  # f32  [rows, HIDDEN] output
    final_bf16_ptr,  # bf16 [rows, HIDDEN] output
    invstd_div_ptr,  # f32  [rows]  output = rsqrt / HIDDEN
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat_bf16 = ct.load(flat_bf16_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x = ct.astype(flat_bf16, ct.float32) + residual

    inv_H = 1.0 / HIDDEN_C
    mean = ct.sum(x) * inv_H
    centered = x - mean
    variance = ct.sum(centered * centered) * inv_H
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(final_bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(invstd_div_ptr, index=(row,), tile=ct.reshape(invstd * inv_H, (1,)))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="ba44cc6a", BLOCK_H=HIDDEN)
def oracle_forward(inputs, *, BLOCK_H):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)  # (8, 512, 4096)
    final_shape = _as_shape(shape1)  # (4096, 4096)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    div_shape = view_shape[:-1] + (1,)

    normalized = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=arg0_1.device, dtype=torch.float32,
    )
    final_bf16 = torch.empty_strided(
        final_shape, _contiguous_stride(final_shape),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        div_shape, _contiguous_stride(div_shape),
        device=arg0_1.device, dtype=torch.float32,
    )

    # Reshape all 3D views to [rows, hidden] for the kernel.
    residual_2d = arg1_1.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    invstd_1d = invstd_div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _add_layernorm_kernel,
        (
            arg0_1, residual_2d, arg2_1, arg3_1,
            normalized_2d, final_bf16, invstd_1d,
            hidden, BLOCK_H,
        ),
    )
    return normalized, final_bf16, invstd_div
