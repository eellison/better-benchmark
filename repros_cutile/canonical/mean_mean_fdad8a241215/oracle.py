"""cuTile port of mean_mean_fdad8a241215: dual bf16 RMSNorm chain (gemma-style).

Note: Second RMSNorm outputs f32 (no bf16 cast), then final bf16 cast to output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _dual_rmsnorm_kernel(
    x_ptr,          # bf16 [rows, HIDDEN]
    weight0_ptr,    # bf16 [HIDDEN]
    residual_ptr,   # bf16 [rows, HIDDEN]
    weight1_ptr,    # bf16 [HIDDEN]
    add_out_ptr,    # bf16 [rows, HIDDEN]
    norm_out_ptr,   # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x_f = ct.astype(x, ct.float32)
    sum_sq0 = ct.sum(x_f * x_f)
    inv_rms0 = ct.rsqrt(sum_sq0 * (1.0 / HIDDEN) + 1.0e-6)
    weight0 = ct.load(weight0_ptr, index=(0,), shape=(BLOCK_H,))
    w0_f = ct.astype(weight0, ct.float32)
    w0_2d = ct.reshape(w0_f, (1, BLOCK_H))
    first = (x_f * inv_rms0) * (w0_2d + 1.0)
    first_bf16 = ct.astype(first, ct.bfloat16)

    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    resid_f = ct.astype(residual, ct.float32)
    first_f = ct.astype(first_bf16, ct.float32)
    add_bf16 = ct.astype(resid_f + first_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=add_bf16)

    add_f = ct.astype(add_bf16, ct.float32)
    sum_sq1 = ct.sum(add_f * add_f)
    inv_rms1 = ct.rsqrt(sum_sq1 * (1.0 / HIDDEN) + 1.0e-6)
    weight1 = ct.load(weight1_ptr, index=(0,), shape=(BLOCK_H,))
    w1_f = ct.astype(weight1, ct.float32)
    w1_2d = ct.reshape(w1_f, (1, BLOCK_H))
    second = (add_f * inv_rms1) * (w1_2d + 1.0)
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(second, ct.bfloat16))


@oracle_impl(hardware="B200", point="ab924692", BLOCK_H=4096)
@oracle_impl(hardware="B200", point="5c119e0a", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    x, weight0, residual, weight1, _shape0, out_shape0, out_shape1 = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])

    add_out = torch.empty_like(residual)
    norm_base = torch.empty_like(residual)

    x_2d = x.view(rows, hidden)
    residual_2d = residual.view(rows, hidden)
    add_2d = add_out.view(rows, hidden)
    norm_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dual_rmsnorm_kernel,
        (x_2d, weight0, residual_2d, weight1, add_2d, norm_2d, hidden, BLOCK_H),
    )
    return (
        add_out,
        norm_base.view(tuple(int(dim) for dim in out_shape0)),
        norm_base.view(tuple(int(dim) for dim in out_shape1)),
    )
