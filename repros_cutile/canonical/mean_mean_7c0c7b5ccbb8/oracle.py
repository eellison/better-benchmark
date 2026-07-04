"""cuTile port of mean_mean_7c0c7b5ccbb8: Gemma dual RMSNorm with residual.

Row kernel: first RMSNorm(x) with weight0+1 affine, bf16 rounded; residual add
bf16-rounded; second RMSNorm on the add, weight1+1 affine, bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _dual_rmsnorm_final_kernel(
    x_ptr,          # bf16 [rows, HIDDEN]
    weight0_ptr,    # bf16 [HIDDEN]
    residual_ptr,   # bf16 [rows, HIDDEN]
    weight1_ptr,    # bf16 [HIDDEN]
    out_ptr,        # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x_f = ct.astype(x, ct.float32)
    sum_sq0 = ct.sum(x_f * x_f)
    inv0 = ct.rsqrt(sum_sq0 * (1.0 / HIDDEN) + 1.0e-6)
    w0 = ct.load(weight0_ptr, index=(0,), shape=(BLOCK_H,))
    w0_2d = ct.reshape(ct.astype(w0, ct.float32), (1, BLOCK_H))
    first_bf16 = ct.astype((x_f * inv0) * (w0_2d + 1.0), ct.bfloat16)

    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    resid_f = ct.astype(residual, ct.float32)
    add_bf16 = ct.astype(resid_f + ct.astype(first_bf16, ct.float32), ct.bfloat16)
    add_f = ct.astype(add_bf16, ct.float32)

    sum_sq1 = ct.sum(add_f * add_f)
    inv1 = ct.rsqrt(sum_sq1 * (1.0 / HIDDEN) + 1.0e-6)
    w1 = ct.load(weight1_ptr, index=(0,), shape=(BLOCK_H,))
    w1_2d = ct.reshape(ct.astype(w1, ct.float32), (1, BLOCK_H))
    out_bf16 = ct.astype((add_f * inv1) * (w1_2d + 1.0), ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out_bf16)


@oracle_impl(hardware="B200", point="ab924692", BLOCK_H=4096)
@oracle_impl(hardware="B200", point="5c119e0a", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    residual_2d = arg2_1.view(rows, hidden)
    out = torch.empty_strided(
        (rows, hidden), (hidden, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _dual_rmsnorm_final_kernel,
        (arg0_1, arg1_1, residual_2d, arg3_1, out, hidden, BLOCK_H),
    )
    return out
