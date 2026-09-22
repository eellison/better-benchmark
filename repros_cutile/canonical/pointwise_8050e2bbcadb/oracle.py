"""cuTile port of pointwise_8050e2bbcadb (NEW_PATTERN): broadcast a fp32 [1,1,768]
row over 128 rows with bf16 cast, and return the flattened [128, 768] bf16."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 768


@ct.kernel
def _broadcast_cast_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)
    # Load row vector of length BLOCK_N from x[0, 0, col_block*BLOCK_N:(col_block+1)*BLOCK_N]
    values = ct.load(x_ptr, index=(col_block,), shape=(BLOCK_N,))
    # Broadcast to (BLOCK_M, BLOCK_N) and cast to bf16
    values_2d = ct.reshape(values, (1, BLOCK_N))
    ones = ct.full(shape=(BLOCK_M, 1), fill_value=1.0, dtype=ct.float32)
    broadcast = ones * values_2d
    ct.store(
        out_ptr,
        index=(row_block, col_block),
        tile=ct.astype(broadcast, ct.bfloat16),
    )


@oracle_impl(hardware="B200", point="1b3ee0df", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    x, _shape_param_0, shape = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # x shape (1, 1, 768); view as (768,) for cuTile load
    x_flat = x.view(HIDDEN)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(ROWS, BLOCK_M), ct.cdiv(HIDDEN, BLOCK_N), 1),
        _broadcast_cast_kernel,
        (x_flat, out, BLOCK_M, BLOCK_N),
    )
    return out
