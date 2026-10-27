"""cuTile port of var_mean_787e1d544efe: BEiT mixed-dtype affine residual LayerNorm.

Row-wise: add(gamma*flat + residual), var_mean over hidden, affine, bf16 cast.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _mixed_affine_residual_layernorm_kernel(
    flat_bf16_ptr,    # bf16 [rows, HIDDEN]
    gamma_ptr,        # f32 [HIDDEN]
    residual_ptr,     # f32 [rows, HIDDEN]
    weight_ptr,       # f32 [HIDDEN]
    bias_ptr,         # f32 [HIDDEN]
    add_out_ptr,      # f32 [rows, HIDDEN]
    norm_out_ptr,     # f32 [rows, HIDDEN]
    final_out_ptr,    # bf16 [rows, HIDDEN]
    invstd_div_ptr,   # f32 [rows]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_bf16_ptr, index=(row, 0), shape=(1, BLOCK_H))
    gamma = ct.load(gamma_ptr, index=(0,), shape=(BLOCK_H,))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    flat_f = ct.astype(flat, ct.float32)
    gamma_f = ct.astype(gamma, ct.float32)
    residual_f = ct.astype(residual, ct.float32)

    gamma_2d = ct.reshape(gamma_f, (1, BLOCK_H))
    x = residual_f + gamma_2d * flat_f
    ct.store(add_out_ptr, index=(row, 0), tile=x)

    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-6)
    norm = centered * invstd
    ct.store(norm_out_ptr, index=(row, 0), tile=norm)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    final_f = norm * weight_2d + bias_2d
    ct.store(final_out_ptr, index=(row, 0), tile=ct.astype(final_f, ct.bfloat16))

    invstd_scalar = ct.full(shape=(1,), fill_value=invstd / HIDDEN, dtype=ct.float32)
    ct.store(invstd_div_ptr, index=(row,), tile=invstd_scalar)


@oracle_impl(hardware="B200", point="f4c82f7a", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in shape0)
    final_shape = tuple(int(dim) for dim in shape1)

    # arg2 is [128,197,768]; residual as [rows, hidden]
    residual_2d = arg2_1.view(rows, hidden)
    add_out = torch.empty_like(residual_2d)
    norm_out = torch.empty_like(residual_2d)
    final_out = torch.empty_strided(
        final_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        (add_shape[0], add_shape[1], 1),
        (add_shape[1], 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    invstd_flat = invstd_div.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _mixed_affine_residual_layernorm_kernel,
        (
            arg0_1,
            arg1_1,
            residual_2d,
            arg3_1,
            arg4_1,
            add_out,
            norm_out,
            final_out,
            invstd_flat,
            hidden,
            BLOCK_H,
        ),
    )
    # Reshape add_out/norm_out to match Repro shapes
    return add_out.view(add_shape), norm_out.view(add_shape), final_out.view(final_shape), invstd_div
