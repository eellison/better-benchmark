"""cuTile port of var_mean_692d2645772d (SCHEDULER_FUSION): DeiT distilled-token
patch LayerNorm. The Triton oracle inlines the class/dist/patch cat + position
add producer into one row kernel. We materialize `cat`/`add` via torch (they
are inputs to the LayerNorm) then run a cuTile LayerNorm row kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
EPS = 1.0e-6


@ct.kernel
def _layernorm_row_kernel(
    add_ptr,        # f32 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    mean_out_ptr,   # f32 [ROWS]
    rsqrt_out_ptr,  # f32 [ROWS]
    final_ptr,      # bf16 [ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.astype(
        ct.load(add_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32
    )
    inv_h = 1.0 / HIDDEN_
    x_sum = ct.sum(x, axis=1, keepdims=True)
    x_sq_sum = ct.sum(x * x, axis=1, keepdims=True)
    mean = x_sum * inv_h
    variance = x_sq_sum * inv_h - mean * mean
    centered = x - mean
    invstd = ct.rsqrt(variance + 1.0e-6)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = centered * invstd * weight_2d + bias_2d
    final = ct.astype(affine, ct.bfloat16)
    ct.store(final_ptr, index=(row, 0), tile=final)

    ct.store(mean_out_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_out_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))


@oracle_impl(hardware="B200", point="1d6386ee", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _s0, _s1, _s2, _s3 = inputs
    view_shape = tuple(int(dim) for dim in _s0)
    expand_shape_1 = tuple(int(dim) for dim in _s1)
    expand_shape_2 = tuple(int(dim) for dim in _s2)
    flat_shape = tuple(int(dim) for dim in _s3)

    # Materialize cat + add via torch.
    view = arg0_1.view(view_shape)
    permute = view.permute(0, 2, 1)
    expand = arg1_1.expand(expand_shape_1)
    expand_1 = arg2_1.expand(expand_shape_2)
    cat = torch.cat([expand, expand_1, permute], dim=1)
    add = cat + arg3_1

    rows = int(add.shape[0]) * int(add.shape[1])
    hidden = int(add.shape[2])
    add_2d = add.view(rows, hidden)
    mean_out = torch.empty(
        (rows,), device=arg0_1.device, dtype=torch.float32
    )
    rsqrt_out = torch.empty(
        (rows,), device=arg0_1.device, dtype=torch.float32
    )
    final = torch.empty(
        flat_shape, device=arg0_1.device, dtype=torch.bfloat16
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layernorm_row_kernel,
        (add_2d, arg4_1, arg5_1, mean_out, rsqrt_out, final, hidden, BLOCK_H),
    )
    mean_out_view = mean_out.view(int(add.shape[0]), int(add.shape[1]), 1)
    rsqrt_out_view = rsqrt_out.view(int(add.shape[0]), int(add.shape[1]), 1)
    return cat, add, mean_out_view, rsqrt_out_view, final
