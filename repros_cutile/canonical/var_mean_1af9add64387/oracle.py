"""cuTile port of var_mean_1af9add64387: residual LayerNorm alias.

bf16 residual add, correction=0 row var/mean, affine, output bf16.
Multiple alias views returned that all share the same underlying storage.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_layernorm_kernel(
    flat_ptr,       # (rows, HIDDEN) bf16
    residual_ptr,   # (rows, HIDDEN) bf16 (viewed as 2D)
    weight_ptr,     # (HIDDEN,) bf16
    bias_ptr,       # (HIDDEN,) bf16
    out_ptr,        # (rows, HIDDEN) bf16
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    add = ct.astype(flat_f + resid_f, ct.bfloat16)
    x = ct.astype(add, ct.float32)
    inv_h = 1.0 / HIDDEN
    mean = ct.sum(x) * inv_h
    mean_sq = ct.sum(x * x) * inv_h
    variance = mean_sq - mean * mean
    invstd = ct.rsqrt(variance + 1.0e-12)
    centered = x - mean
    normalized = centered * invstd
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    y = normalized * weight_2d + bias_2d
    y_bf16 = ct.astype(y, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=y_bf16)


@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3, shape4 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    base_shape = tuple(int(d) for d in shape0)
    out = torch.empty_strided(
        base_shape,
        (base_shape[1] * base_shape[2], base_shape[2], 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    out_2d = out.view(rows, hidden)
    residual_2d = arg1_1.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (rows, 1, 1), _residual_layernorm_kernel,
              (arg0_1, residual_2d, arg2_1, arg3_1, out_2d, hidden, BLOCK_H))
    return (
        out,  # [32, 512, 768]
        out_2d.view(tuple(int(d) for d in shape1)),
        out_2d.view(tuple(int(d) for d in shape2)),
        out_2d.view(tuple(int(d) for d in shape3)),
        out_2d.view(tuple(int(d) for d in shape4)),
        out.permute(0, 2, 1),
    )
