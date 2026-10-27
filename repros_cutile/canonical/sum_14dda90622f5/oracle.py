"""cuTile port of sum_14dda90622f5: Lennard-Jones bf16 producer + column sum.

Compute value = arg0 * (1 - arg1^2) in fp32, cast to bf16, then reduce over
axis 0 producing f32[16] (with bf16 rounding boundary on the reduction output).
Return (out.permute(1,0), sum_out).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 128
N = 16


@ct.kernel
def _materialize_and_sum_kernel(
    arg0_ptr,  # bf16 [M, N]
    arg1_ptr,  # bf16 [M, N]
    out_ptr,   # bf16 [M, N]
    sum_ptr,   # f32 [N]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    x = ct.load(arg0_ptr, index=(0, 0), shape=(BLOCK_M, BLOCK_N))
    y = ct.load(arg1_ptr, index=(0, 0), shape=(BLOCK_M, BLOCK_N))

    x_f = ct.astype(x, ct.float32)
    y_f = ct.astype(y, ct.float32)
    squared = y_f * y_f
    complement = 1.0 - squared
    product = x_f * complement
    value_bf16 = ct.astype(product, ct.bfloat16)
    ct.store(out_ptr, index=(0, 0), tile=value_bf16)

    value_f = ct.astype(value_bf16, ct.float32)
    col_sum = ct.sum(value_f, axis=0)
    col_sum_bf16 = ct.astype(col_sum, ct.bfloat16)
    col_sum_f = ct.astype(col_sum_bf16, ct.float32)
    ct.store(sum_ptr, index=(0,), tile=col_sum_f)


@oracle_impl(hardware="B200", point="387810f3")
def oracle_forward(inputs):
    arg0, arg1, shape = inputs
    del shape

    out = torch.empty_strided((M, N), (N, 1), device=arg0.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((N,), (1,), device=arg0.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _materialize_and_sum_kernel,
        (arg0, arg1, out, sum_out, M, N),
    )
    return out.permute(1, 0), sum_out
