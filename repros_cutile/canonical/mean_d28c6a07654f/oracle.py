"""cuTile port of mean_d28c6a07654f: residual RMSNorm with mm+residual+weight paths and returned aliases.

For each row: bf16 residual add (rounded), fp32 mean-square, rsqrt(eps=1e-6),
bf16 normalize, bf16 weight multiply. Returns (add, norm.view(...), norm.view(...)).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_rmsnorm_kernel(
    mm_ptr,        # bf16 [rows, HIDDEN]
    residual_ptr,  # bf16 [rows, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    add_out_ptr,   # bf16 [rows, HIDDEN]
    norm_out_ptr,  # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    mm = ct.load(mm_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    mm_f = ct.astype(mm, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    add = resid_f + mm_f
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


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="da8b94aa", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="111af936", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    mm, residual, weight, add_shape, out_shape0, out_shape1 = inputs
    rows = int(mm.shape[0])
    hidden = int(mm.shape[1])

    add_shape = tuple(int(d) for d in add_shape)
    out_base = torch.empty_strided(
        add_shape, _contiguous_stride(add_shape), device=mm.device, dtype=torch.bfloat16
    )
    norm_base = torch.empty_strided(
        add_shape, _contiguous_stride(add_shape), device=mm.device, dtype=torch.bfloat16
    )
    # Reshape 3D residual to 2D [rows, hidden] for cuTile access
    residual_2d = residual.view(rows, hidden)
    out_base_2d = out_base.view(rows, hidden)
    norm_base_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_kernel,
        (mm, residual_2d, weight, out_base_2d, norm_base_2d, hidden, BLOCK_H),
    )
    return (
        out_base,
        norm_base.view(tuple(int(d) for d in out_shape0)),
        norm_base.view(tuple(int(d) for d in out_shape1)),
    )
