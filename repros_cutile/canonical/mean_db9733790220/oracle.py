"""cuTile port of mean_db9733790220: residual RMSNorm row kernel.

Ports the Triton `_residual_rmsnorm_kernel` — for each row: bf16 residual add
(rounded to bf16), fp32 mean-square, rsqrt with eps, bf16 normalize, bf16
weight multiply, output bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_rmsnorm_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    residual_ptr,  # bf16 [rows, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    out_ptr,       # bf16 [rows, HIDDEN]
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

    x = ct.astype(add_bf16, ct.float32)
    sum_sq = ct.sum(x * x)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    norm_bf16 = ct.astype(x * inv_rms, ct.bfloat16)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    # Broadcast weight [1, BLOCK_H] against norm [1, BLOCK_H]
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(ct.astype(norm_bf16, ct.float32) * weight_2d, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="da8b94aa", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="111af936", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    flat, residual, weight, _view_shape, out_shape = inputs
    rows = int(flat.shape[0])
    hidden = int(flat.shape[1])
    out = torch.empty_strided(
        tuple(int(dim) for dim in out_shape),
        (hidden, 1),
        device=flat.device,
        dtype=torch.bfloat16,
    )
    # cuTile only supports 1-3D grids and needs contiguous logical views.
    # Reshape the residual (which is [1, rows, hidden] contiguous) to [rows, hidden]
    residual_2d = residual.view(rows, hidden)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_kernel,
        (flat, residual_2d, weight, out, hidden, BLOCK_H),
    )
    return out
