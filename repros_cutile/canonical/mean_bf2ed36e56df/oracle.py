"""cuTile port of mean_bf2ed36e56df: bf16 residual-add RMSNorm with alias views."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_rmsnorm_kernel(
    flat_ptr,       # bf16 [rows, HIDDEN]
    residual_ptr,   # bf16 [rows, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    add_out_ptr,    # bf16 [rows, HIDDEN]
    norm_out_ptr,   # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    added_bf16 = ct.astype(resid_f + flat_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=added_bf16)

    x = ct.astype(added_bf16, ct.float32)
    sum_sq = ct.sum(x * x)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    normalized_bf16 = ct.astype(x * inv_rms, ct.bfloat16)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(weight_2d * ct.astype(normalized_bf16, ct.float32), ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=out)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="841ed042", BLOCK_H=512)
@oracle_impl(hardware="B200", point="40057a60", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, add_shape_arg, shape1, shape2, shape3 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in add_shape_arg)
    stride = _contiguous_stride(add_shape)

    add_out = torch.empty_strided(add_shape, stride, device=arg0_1.device, dtype=torch.bfloat16)
    norm_base = torch.empty_strided(add_shape, stride, device=arg0_1.device, dtype=torch.bfloat16)

    add_2d = add_out.view(rows, hidden)
    norm_2d = norm_base.view(rows, hidden)
    residual_2d = arg1_1.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_kernel,
        (arg0_1, residual_2d, arg2_1, add_2d, norm_2d, hidden, BLOCK_H),
    )
    return (
        add_out,
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
        norm_base.view(tuple(int(dim) for dim in shape3)),
    )
