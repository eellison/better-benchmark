"""cuTile port of var_mean_a82162370df1: GoogleFNet residual LayerNorm.

For each row: add flat + residual (both f32), compute mean/var over HIDDEN=768,
apply rsqrt(eps=1e-12), then affine (weight*normalized + bias), store f32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 768
EPS = 1.0e-12


@ct.kernel
def _residual_layernorm_kernel(
    flat_ptr,       # f32 [ROWS, HIDDEN]
    residual_ptr,   # f32 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    out_ptr,        # f32 [ROWS, HIDDEN]
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    x = flat + residual  # both f32 already
    sum_x = ct.sum(x)
    mean = sum_x * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    out = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=out)


@oracle_impl(
    hardware="B200",
    point="e864a84c",
    BLOCK_H=1024,
)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1 = inputs
    flat = arg0_1  # [16384, 768] f32
    residual_3d = arg1_1  # [32, 512, 768] f32
    residual = residual_3d.view(ROWS, HIDDEN)
    weight = arg2_1
    bias = arg3_1

    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=flat.device,
        dtype=torch.float32,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _residual_layernorm_kernel,
        (flat, residual, weight, bias, out, BLOCK_H),
    )
    return out
