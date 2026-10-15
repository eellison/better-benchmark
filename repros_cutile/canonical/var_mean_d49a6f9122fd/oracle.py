"""cuTile port of var_mean_d49a6f9122fd: XLNet residual LayerNorm + permute clone.

For each of 8192 rows: bf16 flat + bf16 residual, LayerNorm in fp32 (eps=1e-12),
bf16 affine, then permute [512,16,1024] -> [16,512,1024] contiguous.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


OUTER = 512
INNER = 16
HIDDEN = 1024
EPS = 1.0e-12
ROWS = OUTER * INNER


@ct.kernel
def _residual_ln_permute_kernel(
    flat_ptr,      # bf16 (ROWS, HIDDEN)
    residual_ptr,  # bf16 (ROWS, HIDDEN)
    weight_ptr,    # bf16 (HIDDEN,)
    bias_ptr,      # bf16 (HIDDEN,)
    out_ptr,       # bf16 (INNER, OUTER, HIDDEN)  (permuted)
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    resid = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    # Match Repro: add in bf16 first, then LN in fp32.
    x_bf = ct.astype(ct.astype(flat, ct.float32) + ct.astype(resid, ct.float32), ct.bfloat16)
    x = ct.astype(x_bf, ct.float32)

    mean = ct.sum(x, axis=1, keepdims=True) * (1.0 / BLOCK_H)
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * (1.0 / BLOCK_H)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    out_bf = ct.astype(affine, ct.bfloat16)

    # Row layout is [OUTER, INNER]; output layout [INNER, OUTER].
    outer = row // INNER
    inner = row - outer * INNER
    ct.store(out_ptr, index=(inner, outer, 0), tile=ct.reshape(out_bf, (1, 1, BLOCK_H)))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="c4c9bec6", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    out_shape = _as_shape(shape1)  # (8192, 1024)
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out_3d = out.view(INNER, OUTER, HIDDEN)
    resid_2d = arg1_1.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _residual_ln_permute_kernel,
        (arg0_1, resid_2d, arg2_1, arg3_1, out_3d, HIDDEN),
    )
    return out
