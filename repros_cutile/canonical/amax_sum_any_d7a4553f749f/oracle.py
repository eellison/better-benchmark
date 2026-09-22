"""cuTile port of amax_sum_any_d7a4553f749f: bf16 softmax with all-inf fallback.

Row-tiled softmax on bf16 [n_rows, k_len]. For rows with any finite entry
computes stable softmax; for all -inf rows, returns fallback tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_softmax_fallback_kernel(
    x_ptr, fallback_ptr, out_ptr,
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    x = ct.load(x_ptr, index=(row_tile, 0), shape=(BLOCK_M, BLOCK_N))
    scores = ct.astype(x, ct.float32)
    neg_inf = ct.full(shape=(BLOCK_M, BLOCK_N),
                      fill_value=float("-inf"), dtype=ct.float32)
    live = scores != neg_inf
    ones = ct.full(shape=(BLOCK_M, BLOCK_N), fill_value=1, dtype=ct.int32)
    zeros_i = ct.full(shape=(BLOCK_M, BLOCK_N), fill_value=0, dtype=ct.int32)
    has_any_i = ct.max(ct.where(live, ones, zeros_i), axis=1, keepdims=True)
    has_any = has_any_i != ct.full(shape=(BLOCK_M, 1),
                                    fill_value=0, dtype=ct.int32)

    row_max = ct.max(scores, axis=1, keepdims=True)
    zero_scalar = ct.full(shape=(BLOCK_M, 1), fill_value=0.0, dtype=ct.float32)
    safe_max = ct.where(has_any, row_max, zero_scalar)
    zeros_f = ct.full(shape=(BLOCK_M, BLOCK_N),
                       fill_value=0.0, dtype=ct.float32)
    numer = ct.exp(scores - safe_max)
    numer = ct.where(live, numer, zeros_f)
    denom = ct.sum(numer, axis=1, keepdims=True)
    one_scalar = ct.full(shape=(BLOCK_M, 1), fill_value=1.0, dtype=ct.float32)
    denom = ct.where(has_any, denom, one_scalar)
    probs = numer / denom

    fallback = ct.astype(
        ct.load(fallback_ptr, index=(row_tile, 0),
                shape=(BLOCK_M, BLOCK_N)),
        ct.float32,
    )
    out = ct.where(has_any, probs, fallback)
    ct.store(out_ptr, index=(row_tile, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="b31e9601", BLOCK_M=4)
def oracle_forward(inputs, *, BLOCK_M: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    out_3d_shape = tuple(int(dim) for dim in _shape_param_2)
    k_len = int(arg0_1.shape[-1])
    n_rows = arg0_1.numel() // k_len

    out_3d = torch.empty_strided(
        out_3d_shape,
        (out_3d_shape[1] * out_3d_shape[2], out_3d_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    x_flat = arg0_1.view(n_rows, k_len)
    fallback_flat = arg1_1.view(n_rows, k_len)
    out_flat = out_3d.view(n_rows, k_len)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows // BLOCK_M, 1, 1), _bf16_softmax_fallback_kernel,
        (x_flat, fallback_flat, out_flat, BLOCK_M, k_len),
    )
    return out_3d.view_as(arg1_1), out_3d
