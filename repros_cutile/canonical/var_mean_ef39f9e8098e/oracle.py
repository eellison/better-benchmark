"""cuTile port of var_mean_ef39f9e8098e (SCHEDULER_FUSION): DINOv2 affine-residual
LayerNorm. Returns (add, mul_1, view_1, div).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768


@ct.kernel
def _dinov2_layernorm_kernel(
    flat_ptr,        # bf16 [ROWS, HIDDEN]
    scale_ptr,       # f32 [HIDDEN]
    residual_ptr,    # f32 [ROWS, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    add_out_ptr,     # f32 [ROWS, HIDDEN]
    mul1_out_ptr,    # f32 [ROWS, HIDDEN]
    final_ptr,       # bf16 [ROWS, HIDDEN]
    div_ptr,         # f32 [ROWS]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.astype(
        ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32
    )
    scale = ct.astype(
        ct.load(scale_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32
    )
    residual = ct.astype(
        ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32
    )
    scale_2d = ct.reshape(scale, (1, BLOCK_H))
    mul_val = flat * scale_2d
    add = residual + mul_val
    ct.store(add_out_ptr, index=(row, 0), tile=add)

    inv_h = 1.0 / HIDDEN_
    x_sum = ct.sum(add, axis=1, keepdims=True)
    x_sq_sum = ct.sum(add * add, axis=1, keepdims=True)
    mean = x_sum * inv_h
    variance = x_sq_sum * inv_h - mean * mean
    invstd = ct.rsqrt(variance + 1.0e-6)
    centered = add - mean
    mul1 = centered * invstd
    ct.store(mul1_out_ptr, index=(row, 0), tile=mul1)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = mul1 * weight_2d + bias_2d
    final = ct.astype(affine, ct.bfloat16)
    ct.store(final_ptr, index=(row, 0), tile=final)

    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd, (1,)) * (1.0 / HIDDEN_))


@oracle_impl(hardware="B200", point="70d2be15", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _s0, _s1 = inputs
    view_shape = tuple(int(dim) for dim in _s0)
    flat_shape = tuple(int(dim) for dim in _s1)
    rows = view_shape[0] * view_shape[1]
    hidden = int(arg0_1.shape[1])

    flat_2d = arg0_1.view(rows, hidden)
    residual_2d = arg2_1.view(rows, hidden)

    add_out = torch.empty(
        view_shape, device=arg0_1.device, dtype=torch.float32
    )
    mul1_out = torch.empty(
        view_shape, device=arg0_1.device, dtype=torch.float32
    )
    final = torch.empty(
        flat_shape, device=arg0_1.device, dtype=torch.bfloat16
    )
    div = torch.empty(
        (view_shape[0], view_shape[1], 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    add_out_2d = add_out.view(rows, hidden)
    mul1_out_2d = mul1_out.view(rows, hidden)
    div_flat = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dinov2_layernorm_kernel,
        (
            flat_2d,
            arg1_1,
            residual_2d,
            arg3_1,
            arg4_1,
            add_out_2d,
            mul1_out_2d,
            final,
            div_flat,
            hidden,
            BLOCK_H,
        ),
    )
    return add_out, mul1_out, final, div
