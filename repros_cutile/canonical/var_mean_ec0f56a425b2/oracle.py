"""cuTile port of var_mean_ec0f56a425b2: GPT-J mixed-dtype residual LayerNorm.

For each row: bf16(a+b), then f32(bf16_ab + f32_residual), var_mean, rsqrt,
normalize, affine, bf16 output. Returns [add_out, norm_out, final_out, invstd/H].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 4096


@ct.kernel
def _gptj_mixed_residual_layernorm_kernel(
    lhs_ptr,        # bf16 [rows, HIDDEN]
    rhs_ptr,        # bf16 [rows, HIDDEN]
    residual_ptr,   # f32  [rows, HIDDEN]
    weight_ptr,     # f32  [HIDDEN]
    bias_ptr,       # f32  [HIDDEN]
    add_out_ptr,    # f32  [rows, HIDDEN]
    norm_out_ptr,   # f32  [rows, HIDDEN]
    final_out_ptr,  # bf16 [rows, HIDDEN]
    invstd_div_ptr, # f32  [rows]
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    lhs = ct.load(lhs_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    # bf16 rounding of (lhs + rhs), then + residual in fp32
    ab_bf16 = ct.astype(ct.astype(lhs, ct.float32) + ct.astype(rhs, ct.float32), ct.bfloat16)
    x = ct.astype(ab_bf16, ct.float32) + residual
    ct.store(add_out_ptr, index=(row, 0), tile=x)

    inv_h = 1.0 / HIDDEN_C
    mean = ct.sum(x) * inv_h
    centered = x - mean
    var = ct.sum(centered * centered) * inv_h
    invstd = ct.rsqrt(var + 1.0e-5)
    norm = centered * invstd
    ct.store(norm_out_ptr, index=(row, 0), tile=norm)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    final = ct.astype(norm * weight_2d + bias_2d, ct.bfloat16)
    ct.store(final_out_ptr, index=(row, 0), tile=final)
    # invstd/H is a scalar per row.
    invstd_div = ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + invstd * (1.0 / HIDDEN_C), (1,))
    ct.store(invstd_div_ptr, index=(row,), tile=invstd_div)


@oracle_impl(hardware="B200", point="3fdaed2a", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape1, shape2 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    final_shape = tuple(int(dim) for dim in shape2)

    add_out = torch.empty_like(arg2_1)
    norm_out = torch.empty_like(arg2_1)
    final_out = torch.empty_strided(
        final_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty(
        (1, rows, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    residual_2d = arg2_1.view(rows, hidden)
    add_out_2d = add_out.view(rows, hidden)
    norm_out_2d = norm_out.view(rows, hidden)
    invstd_flat = invstd_div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _gptj_mixed_residual_layernorm_kernel,
        (arg0_1, arg1_1, residual_2d, arg3_1, arg4_1,
         add_out_2d, norm_out_2d, final_out, invstd_flat, hidden, BLOCK_H),
    )
    return add_out, norm_out, final_out, invstd_div
