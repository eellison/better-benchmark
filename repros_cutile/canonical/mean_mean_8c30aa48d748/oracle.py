"""cuTile port of mean_mean_8c30aa48d748: dual-RMSNorm bf16 row kernel.

Chains two RMSNorm rows: first normalizes `arg0` with `weight0`, adds to
`residual` (bf16-rounded), then normalizes that add with `weight1`. Emits three
aliased views of the second buffer.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _dual_rmsnorm_bf16_kernel(
    x0_ptr,          # bf16 [rows, HIDDEN]
    weight0_ptr,     # bf16 [HIDDEN]
    residual_ptr,    # bf16 [rows, HIDDEN]
    weight1_ptr,     # bf16 [HIDDEN]
    add_out_ptr,     # bf16 [rows, HIDDEN]
    norm_out_ptr,    # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x0_bf16 = ct.load(x0_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x0 = ct.astype(x0_bf16, ct.float32)
    sum0 = ct.sum(x0 * x0)
    inv0 = ct.rsqrt(sum0 * (1.0 / HIDDEN) + 1.0e-6)

    w0 = ct.load(weight0_ptr, index=(0,), shape=(BLOCK_H,))
    w0_f = ct.astype(w0, ct.float32)
    w0_2d = ct.reshape(w0_f, (1, BLOCK_H))
    norm0_bf16 = ct.astype((x0 * inv0) * (w0_2d + 1.0), ct.bfloat16)

    residual_bf16 = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.astype(residual_bf16, ct.float32)
    add_bf16 = ct.astype(residual_f + ct.astype(norm0_bf16, ct.float32), ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=add_bf16)

    x1 = ct.astype(add_bf16, ct.float32)
    sum1 = ct.sum(x1 * x1)
    inv1 = ct.rsqrt(sum1 * (1.0 / HIDDEN) + 1.0e-6)

    w1 = ct.load(weight1_ptr, index=(0,), shape=(BLOCK_H,))
    w1_f = ct.astype(w1, ct.float32)
    w1_2d = ct.reshape(w1_f, (1, BLOCK_H))
    norm1_bf16 = ct.astype((x1 * inv1) * (w1_2d + 1.0), ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=norm1_bf16)


@oracle_impl(hardware="B200", point="ab924692", BLOCK_H=4096)
@oracle_impl(hardware="B200", point="5c119e0a", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in shape0)

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
    # cuTile: reshape residual [1,rows,hidden] as [rows, hidden] contiguous
    residual_2d = arg2_1.view(rows, hidden)
    add_out_2d = add_out.view(rows, hidden)
    norm_base_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dual_rmsnorm_bf16_kernel,
        (arg0_1, arg1_1, residual_2d, arg3_1, add_out_2d, norm_base_2d, hidden, BLOCK_H),
    )
    return (
        add_out,
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
        norm_base.view(tuple(int(dim) for dim in shape3)),
    )
