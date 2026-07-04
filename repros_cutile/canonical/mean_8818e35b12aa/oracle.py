"""cuTile port of mean_8818e35b12aa: bf16 residual-add + RMSNorm row kernel.

For each row: bf16 residual + flat add (rounded to bf16), fp32 mean-square,
rsqrt with 1e-6 eps, bf16 normalize + weight multiply, bf16 output.
Returns both the residual-add tensor and the final view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_rmsnorm_bf16_kernel(
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

    added_bf16 = ct.astype(ct.astype(residual, ct.float32) + ct.astype(flat, ct.float32), ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=added_bf16)

    x = ct.astype(added_bf16, ct.float32)
    sum_sq = ct.sum(x * x)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    norm_bf16 = ct.astype(x * inv_rms, ct.bfloat16)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(weight_2d * ct.astype(norm_bf16, ct.float32), ct.bfloat16)
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
    arg0_1, arg1_1, arg2_1, add_shape_arg, out_shape_arg = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in add_shape_arg)

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # Reshape residual/add_out/norm_base to [rows, hidden] for cuTile 2D tiles.
    residual_2d = arg1_1.view(rows, hidden)
    add_out_2d = add_out.view(rows, hidden)
    norm_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_bf16_kernel,
        (arg0_1, residual_2d, arg2_1, add_out_2d, norm_2d, hidden, BLOCK_H),
    )
    return add_out, norm_base.view(tuple(int(dim) for dim in out_shape_arg))
