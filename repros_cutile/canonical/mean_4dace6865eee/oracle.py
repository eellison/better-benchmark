"""cuTile port of mean_4dace6865eee: T5 bf16 residual RMSNorm with alias-view returns.

For each row: bf16 residual add rounded to bf16, fp32 mean-square RMS with
eps=1e-6, bf16 normalize, bf16 weight multiply, output bf16. Returns 13 views
of the same base normalized tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_rmsnorm_kernel(
    flat_ptr,      # bf16 (rows, HIDDEN)
    residual_ptr,  # bf16 (rows, HIDDEN)
    weight_ptr,    # bf16 (HIDDEN,)
    out_ptr,       # bf16 (rows, HIDDEN)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    add_bf16 = ct.astype(resid_f + flat_f, ct.bfloat16)
    x = ct.astype(add_bf16, ct.float32)
    sum_sq = ct.sum(x * x)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    norm_bf16 = ct.astype(x * inv_rms, ct.bfloat16)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(ct.astype(norm_bf16, ct.float32) * weight_2d, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="40057a60", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        arg0_1, arg1_1, arg2_1, base_shape_arg,
        shape1, shape2, shape3, shape4, shape5, shape6,
        shape7, shape8, shape9, shape10, shape11, shape12,
    ) = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    base_shape = tuple(int(dim) for dim in base_shape_arg)

    norm_base = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # arg1_1 is [8, 1024, 512] contiguous — view as (rows, hidden) for the kernel
    residual_2d = arg1_1.view(rows, hidden)
    norm_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_kernel,
        (arg0_1, residual_2d, arg2_1, norm_2d, hidden, BLOCK_H),
    )
    return (
        norm_base,
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
        norm_base.view(tuple(int(dim) for dim in shape3)),
        norm_base.view(tuple(int(dim) for dim in shape4)),
        norm_base.view(tuple(int(dim) for dim in shape5)),
        norm_base.view(tuple(int(dim) for dim in shape6)),
        norm_base.view(tuple(int(dim) for dim in shape7)),
        norm_base.view(tuple(int(dim) for dim in shape8)),
        norm_base.view(tuple(int(dim) for dim in shape9)),
        norm_base.view(tuple(int(dim) for dim in shape10)),
        norm_base.view(tuple(int(dim) for dim in shape11)),
        norm_base.view(tuple(int(dim) for dim in shape12)),
    )
