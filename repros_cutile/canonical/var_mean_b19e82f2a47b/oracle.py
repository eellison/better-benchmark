"""cuTile port of var_mean_b19e82f2a47b: residual-add LayerNorm many models.

SCHEDULER_FUSION: for each row, load bf16 x0 + bf16 x1 fp32-add, round to
bf16 side output, compute fp32 mean and variance (correction=0), rsqrt with
eps=1e-5 affine epilogue with per-feature weight+bias, store the bf16 output.
Return the bf16 add side output and three alias views of the norm buffer.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_layernorm_bf16_kernel(
    x0_ptr,
    x1_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x0 = ct.load(x0_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x1 = ct.load(x1_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x0_f = ct.astype(x0, ct.float32)
    x1_f = ct.astype(x1, ct.float32)
    add_bf16 = ct.astype(x1_f + x0_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=add_bf16)

    x = ct.astype(add_bf16, ct.float32)
    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    var = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(var + 1.0e-5)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    y = centered * invstd * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="c4bf51cc", BLOCK_H=4096)
@oracle_impl(hardware="B200", point="7f824027", BLOCK_H=4096)
@oracle_impl(hardware="B200", point="f2c837cd", BLOCK_H=2048)
@oracle_impl(hardware="B200", point="17affd46", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="0b3dc49f", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="ceab07f0", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="a3e95c29", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="f2e11670", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="324149d9", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, shape1, shape2, shape3 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_out = torch.empty_like(arg1_1)
    norm_base = torch.empty_like(arg1_1)

    # arg1_1 may be shape [B, S, H] contiguous. View as [rows, hidden].
    x1_2d = arg1_1.view(rows, hidden)
    add_out_2d = add_out.view(rows, hidden)
    norm_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_bf16_kernel,
        (arg0_1, x1_2d, arg2_1, arg3_1, add_out_2d, norm_2d, hidden, BLOCK_H),
    )
    return (
        add_out,
        norm_base.view(tuple(shape1)),
        norm_base.view(tuple(shape2)),
        norm_base.view(tuple(shape3)),
    )
