"""cuTile port of var_mean_60f28772f7d2: XLNet residual LayerNorm row kernel.

For each row: bf16 add (arg1[row] = residual, arg0[row] = flat), fp32 var_mean,
rsqrt with 1e-12 eps, bf16 affine output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _xlnet_residual_layernorm_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    residual_ptr,  # bf16 [rows, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    bias_ptr,      # bf16 [HIDDEN]
    out_ptr,       # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    # Eager repro: add = residual + flat (bf16), then bf16->f32. So the add is
    # bf16-rounded before the var/mean.
    x_bf16 = ct.astype(ct.astype(flat, ct.float32) + ct.astype(residual, ct.float32), ct.bfloat16)
    x = ct.astype(x_bf16, ct.float32)
    inv_h = 1.0 / HIDDEN
    mean = ct.sum(x) * inv_h
    centered = x - mean
    variance = ct.sum(centered * centered) * inv_h
    invstd = ct.rsqrt(variance + 1.0e-12)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))

    y = centered * invstd * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="c4c9bec6", BLOCK_H=1024, XBLOCK=4)
def oracle_forward(inputs, *, BLOCK_H, XBLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, shape2 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out = torch.empty_like(arg1_1)
    # arg1_1 is [512, 16, 1024]; reshape to [rows, hidden]
    residual_2d = arg1_1.view(rows, hidden)
    out_2d = out.view(rows, hidden)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _xlnet_residual_layernorm_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, out_2d, hidden, BLOCK_H),
    )
    return out, out.view(tuple(shape2))
