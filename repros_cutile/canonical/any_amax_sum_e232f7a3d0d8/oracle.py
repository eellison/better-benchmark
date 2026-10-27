"""cuTile port of any_amax_sum_e232f7a3d0d8: bf16 safe row softmax.

For each row: if all values are -inf, output zeros; else stable softmax.
Uses BLOCK_M rows per block to match Triton's grid.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_safe_softmax_kernel(
    x_ptr,   # bf16 [n_rows, k_len]
    out_ptr,
    n_rows_c: ct.Constant[int],
    k_len_c: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    block = ct.bid(0)
    # k_len equals BLOCK_N in this port (Triton also uses BLOCK_N == k_len).
    x = ct.load(x_ptr, index=(block, 0), shape=(BLOCK_M, BLOCK_N))
    x_f = ct.astype(x, ct.float32)
    row_max = ct.max(x_f, axis=1, keepdims=True)  # [BLOCK_M, 1]
    # has_any: True if any element is not -inf
    neg_inf_val = float("-inf")
    is_live = x_f != neg_inf_val  # bool tile
    # count live per row via sum of int cast
    live_count = ct.sum(ct.astype(is_live, ct.float32), axis=1, keepdims=True)
    has_any = live_count > 0.0  # [BLOCK_M, 1] bool

    safe_max = ct.where(has_any, row_max, ct.zeros((BLOCK_M, 1), dtype=ct.float32))
    numer = ct.exp(x_f - safe_max)
    numer = ct.where(is_live, numer,
                     ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))
    denom = ct.sum(numer, axis=1, keepdims=True)
    denom = ct.where(has_any, denom, ct.full((BLOCK_M, 1), 1.0, dtype=ct.float32))
    probs = numer * (1.0 / denom)
    probs = ct.where(ct.broadcast_to(has_any, (BLOCK_M, BLOCK_N)),
                     probs,
                     ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))
    ct.store(out_ptr, index=(block, 0), tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="9135f859", BLOCK_M=4, BLOCK_N=512)
@oracle_impl(hardware="B200", point="59a0e132", BLOCK_M=4, BLOCK_N=512)
@oracle_impl(hardware="B200", point="730342a3", BLOCK_M=4, BLOCK_N=512)
@oracle_impl(hardware="B200", point="e6f344ac", BLOCK_M=8, BLOCK_N=128)
@oracle_impl(hardware="B200", point="2fe3efd1", BLOCK_M=8, BLOCK_N=128)
@oracle_impl(hardware="B200", point="bcf6fe02", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    x, _shape0, _shape1, _shape2, out_shape = inputs
    k_len = int(x.shape[-1])
    n_rows = x.numel() // k_len
    out_shape = tuple(int(dim) for dim in out_shape)
    out_stride = (out_shape[1] * out_shape[2], out_shape[2], 1)
    out = torch.empty_strided(
        out_shape,
        out_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )
    x_2d = x.view(n_rows, k_len)
    out_2d = out.view(n_rows, k_len)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _bf16_safe_softmax_kernel,
        (x_2d, out_2d, n_rows, k_len, BLOCK_M, BLOCK_N),
    )
    return out
