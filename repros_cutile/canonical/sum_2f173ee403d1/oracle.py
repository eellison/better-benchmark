"""cuTile port of sum_2f173ee403d1: row product-sum with fma epilogue (bf16 out).

Ports the Triton `_row_sum_fma_kernel` — for each row of `[N_ROWS, K]`:
  product = x * grad (fp32)
  row_sum = sum(product, axis=-1)
  out = fma(-grad, row_sum, product) -> bf16
K is a power of 2 (128 or 512), so a single tile per row (BLOCK_K=K) works.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_sum_fma_kernel(
    x_ptr,      # bf16 [N_ROWS, K]
    grad_ptr,   # bf16 [N_ROWS, K]
    out_ptr,    # bf16 [N_ROWS, K]
    BLOCK_M: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(BLOCK_M, BLOCK_K))
    grad = ct.load(grad_ptr, index=(row, 0), shape=(BLOCK_M, BLOCK_K))

    x_f = ct.astype(x, ct.float32)
    grad_f = ct.astype(grad, ct.float32)
    product = x_f * grad_f
    row_sum = ct.sum(product, axis=1, keepdims=True)  # (BLOCK_M, 1)
    out = product - grad_f * row_sum  # equivalent to fma(-grad, row_sum, product)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="0c938b20", BLOCK_M=4, BLOCK_K=128)
@oracle_impl(hardware="B200", point="b31e9601", BLOCK_M=2, BLOCK_K=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_K: int):
    arg0_1, arg1_1, _shape_param_0, shape_param_1 = inputs
    out_shape = _shape_tuple(shape_param_1)
    k = int(arg1_1.shape[-1])
    rows = arg1_1.numel() // k

    out = torch.empty_strided(
        out_shape,
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # cuTile requires flat 2D tile-view. View both x and grad as [N_ROWS, K].
    x2 = arg0_1.view(rows, k)
    grad2 = arg1_1.view(rows, k)
    out2 = out.view(rows, k)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows // BLOCK_M, 1, 1),
        _row_sum_fma_kernel,
        (x2, grad2, out2, BLOCK_M, BLOCK_K),
    )
    return out
