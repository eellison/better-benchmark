"""cuTile port of amax_sum_7f67e161bd21 (ALGEBRAIC_ELIMINATION): bf16 row softmax.

Multi-row: BLOCK_M=16 rows per block to match Triton.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_COLS = 512


@ct.kernel
def _identity_mask_softmax_kernel(
    x_ptr,     # bf16 [n_rows, N_COLS]
    out_ptr,   # bf16 [n_rows, N_COLS]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    block = ct.bid(0)
    x = ct.load(x_ptr, index=(block, 0), shape=(BLOCK_M, BLOCK_N))
    x_f = ct.astype(x, ct.float32)
    row_max = ct.max(x_f, axis=1, keepdims=True)
    numer = ct.exp(x_f - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    out = numer * (1.0 / denom)
    ct.store(out_ptr, index=(block, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="4f884edd", BLOCK_M=16, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    x, _mask_shape, _view_shape, _out_shape = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_rows = x.numel() // int(x.shape[-1])
    x_2d = x.view(n_rows, N_COLS)
    out_2d = out.view(n_rows, N_COLS)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _identity_mask_softmax_kernel,
        (x_2d, out_2d, BLOCK_M, BLOCK_N),
    )
    return out
