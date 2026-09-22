"""cuTile port of pointwise_edf2e0dc44ea (NEW_PATTERN): embedding gather —
copy rows of a f32 weight table into an output indexed by int64 tokens."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _embedding_gather_kernel(
    weight_ptr,
    index_ptr,
    out_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)
    token_ids = ct.load(index_ptr, index=(row_block,), shape=(BLOCK_M,))
    cols = ct.arange(BLOCK_D, dtype=ct.int32) + col_block * BLOCK_D
    cols64 = ct.astype(cols, ct.int64)
    row_indices = ct.reshape(token_ids, (BLOCK_M, 1))
    col_indices = ct.reshape(cols64, (1, BLOCK_D))
    values = ct.gather(weight_ptr, (row_indices, col_indices))
    ct.store(out_ptr, index=(row_block, col_block), tile=values)


# 154a9c83: (T([128,2560], f32), T([128], i64))
@oracle_impl(hardware="B200", point="154a9c83", BLOCK_M=1, BLOCK_D=1024)
# 8ea03555: (T([1024,1024], f32), T([128], i64))
@oracle_impl(hardware="B200", point="8ea03555", BLOCK_M=2, BLOCK_D=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_D: int):
    weight, indices = inputs
    rows = int(indices.numel())
    hidden = int(weight.shape[1])
    output = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=weight.device,
        dtype=weight.dtype,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), ct.cdiv(hidden, BLOCK_D), 1),
        _embedding_gather_kernel,
        (weight, indices, output, hidden, BLOCK_M, BLOCK_D),
    )
    return output
