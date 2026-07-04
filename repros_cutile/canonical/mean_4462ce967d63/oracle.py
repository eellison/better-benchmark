"""cuTile port of mean_4462ce967d63: T5 residual RMSNorm + final scale."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


FINAL_SCALE = 0.04419417382415922


@ct.kernel
def _residual_rmsnorm_scaled_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    HIDDEN: ct.Constant[int],
    FINAL_SCALE_VALUE: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.astype(ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    residual = ct.astype(ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    added_bf16 = ct.astype(residual + flat, ct.bfloat16)
    x = ct.astype(added_bf16, ct.float32)

    square_sum = ct.sum(x * x)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN) + 1.0e-6)
    normalized_bf16 = ct.astype(x * inv_rms, ct.bfloat16)

    weight = ct.reshape(
        ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32),
        (1, BLOCK_H),
    )
    weighted_bf16 = ct.astype(weight * ct.astype(normalized_bf16, ct.float32), ct.bfloat16)
    out = ct.astype(ct.astype(weighted_bf16, ct.float32) * FINAL_SCALE_VALUE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="40057a60", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, base_shape_arg, out_shape_arg = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    base_shape = tuple(int(dim) for dim in base_shape_arg)
    out_shape = tuple(int(dim) for dim in out_shape_arg)

    out_base = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # Flatten the residual (which is (8, 1024, 512) contig) to (rows, hidden)
    residual_2d = arg1_1.view(rows, hidden)
    out_2d = out_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_scaled_kernel,
        (arg0_1, residual_2d, arg2_1, out_2d, hidden, FINAL_SCALE, BLOCK_H),
    )
    return out_base.view(out_shape)
