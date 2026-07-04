"""cuTile port of amax_sum_fa4cc85fe5ad: T5/MT5 bf16 row softmax.

Mirrors the Triton kernel: BLOCK_M rows per program, row-wise softmax over
N_COLS with fp32 accumulation and bf16 output cast.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_softmax_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: ct.Constant[int],
    N_COLS: ct.Constant[int],
):
    row_block = ct.bid(0)
    scores = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, N_COLS))
    scores_f = ct.astype(scores, ct.float32)
    row_max = ct.max(scores_f, axis=1, keepdims=True)
    numer = ct.exp(scores_f - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="215d1cc0", BLOCK_M=16, BLOCK_N=128)
@oracle_impl(hardware="B200", point="5b248c57", BLOCK_M=1, BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_2),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_cols = int(x.shape[-1])
    n_rows = x.numel() // n_cols
    x_2d = x.reshape(n_rows, n_cols)
    out_2d = out.view(n_rows, n_cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _bf16_softmax_kernel,
        (x_2d, out_2d, BLOCK_M, n_cols),
    )
    return out
