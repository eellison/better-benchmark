"""cuTile port of pointwise_299ce2ae2b07: M2M100 computed-index embedding gather.

Row indices are computed from cumsum*mask + 1 (bounded by vocab); each row
gathers HIDDEN=1024 embedding values from the fp32 table.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS0 = 64
ROWS1 = 128
TOKENS = ROWS0 * ROWS1
VOCAB = 1026
HIDDEN = 1024


@ct.kernel
def _computed_embedding_gather_kernel(
    cumsum_ptr,    # int64 [TOKENS]
    mask_ptr,      # int32 [TOKENS]
    table_ptr,     # f32 [VOCAB, HIDDEN]
    out_ptr,       # f32 [TOKENS, HIDDEN]
    BLOCK_N: ct.Constant[int],
):
    tok = ct.bid(0)  # one token row per block
    # Load the one token's cumsum + mask
    cumsum_i64 = ct.load(cumsum_ptr, index=(tok,), shape=(1,))
    mask_i32 = ct.load(mask_ptr, index=(tok,), shape=(1,))
    # Compute row = cumsum.to(int32) * mask (as int64) + 1
    cumsum_i32 = ct.astype(cumsum_i64, ct.int32)
    prod_i32 = cumsum_i32 * mask_i32
    row_i64 = ct.astype(prod_i32, ct.int64) + ct.full(shape=(1,), fill_value=1, dtype=ct.int64)

    # Gather one row from the table: rows shape (1,1), cols shape (1, BLOCK_N)
    row_2d = ct.reshape(row_i64, (1, 1))
    col_offsets = ct.arange(BLOCK_N, dtype=ct.int64)
    cols_2d = ct.reshape(col_offsets, (1, BLOCK_N))
    values = ct.gather(table_ptr, (row_2d, cols_2d))
    ct.store(out_ptr, index=(tok, 0), tile=values)


@oracle_impl(hardware="B200", point="01e90f6e", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    cumsum, mask, table, _shape = inputs
    out = torch.empty_strided(
        (ROWS0, ROWS1, HIDDEN),
        (ROWS1 * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=table.dtype,
    )
    # Flatten both 1D indices and 2D output to (TOKENS,) and (TOKENS, HIDDEN)
    cumsum_1d = cumsum.view(TOKENS)
    mask_1d = mask.view(TOKENS)
    out_2d = out.view(TOKENS, HIDDEN)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (TOKENS, 1, 1),
        _computed_embedding_gather_kernel,
        (cumsum_1d, mask_1d, table, out_2d, BLOCK_N),
    )
    return out
