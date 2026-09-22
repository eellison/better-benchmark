"""cuTile port of mean_c033ad4c69ca: residual RMSNorm returning add and 3 flattened views.

Ports the Triton `_residual_rmsnorm_bf16_kernel` — for each row: bf16 add,
fp32 mean-square, rsqrt with eps, bf16 normalize, bf16 weight multiply.
The returned add tensor is bf16; the normalized output has three view aliases.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_rmsnorm_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    residual_ptr,  # bf16 [rows, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    add_out_ptr,   # bf16 [rows, HIDDEN]
    norm_out_ptr,  # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    add = resid_f + flat_f
    add_bf16 = ct.astype(add, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=add_bf16)

    x = ct.astype(add_bf16, ct.float32)
    sum_sq = ct.sum(x * x)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    norm_bf16 = ct.astype(x * inv_rms, ct.bfloat16)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(ct.astype(norm_bf16, ct.float32) * weight_2d, ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="da8b94aa", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="111af936", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, add_shape_param, shape1, shape2, shape3 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in add_shape_param)

    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    residual_2d = arg1_1.view(rows, hidden)
    add_view_2d = add_out.view(rows, hidden)
    norm_view_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_kernel,
        (arg0_1, residual_2d, arg2_1, add_view_2d, norm_view_2d, hidden, BLOCK_H),
    )
    return (
        add_out,
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
        norm_base.view(tuple(int(dim) for dim in shape3)),
    )
