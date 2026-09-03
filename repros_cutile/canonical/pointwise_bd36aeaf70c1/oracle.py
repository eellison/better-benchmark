"""cuTile port of pointwise_bd36aeaf70c1: OPT id+2 shifted-index side output and embedding gather.

The table has HIDDEN=768 columns which isn't a power of two, so we tile the
column dimension with BLOCK_N=256 (768 = 256 * 3) and gather 3 sub-tiles per row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
SEQ_LEN = 2048
HIDDEN = 768
TOKENS = BATCH * SEQ_LEN


@ct.kernel
def _embedding_shifted_gather_kernel(
    ids_ptr,       # int64 [TOKENS]
    table_ptr,     # f32 [2050, HIDDEN]
    add_out_ptr,   # int64 [TOKENS]
    emb_out_ptr,   # f32 [TOKENS, HIDDEN]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    tok = ct.bid(0)  # in units of BLOCK_M
    col = ct.bid(1)  # in units of BLOCK_N
    ids = ct.load(ids_ptr, index=(tok,), shape=(BLOCK_M,))
    shifted = ids + ct.full(shape=(BLOCK_M,), fill_value=2, dtype=ct.int64)
    # Only store the shifted ids on the first col block (col==0)
    # Simpler: always store — same value written 3x is fine since writes go to same offset.
    ct.store(add_out_ptr, index=(tok,), tile=shifted)
    # Load a tile from the embedding table: rows = shifted, cols = [col*BLOCK_N, (col+1)*BLOCK_N)
    # Use ct.load with 2D shape and a starting column
    values = ct.load(table_ptr, index=(0, col), shape=(1, BLOCK_N))  # placeholder
    # Doesn't work because we need row-indirect load. Use gather.
    # Actually gather needs int index arrays.
    rows_2d = ct.reshape(shifted, (BLOCK_M, 1))
    col_offsets = ct.arange(BLOCK_N, dtype=ct.int64) + col * BLOCK_N
    cols_2d = ct.reshape(col_offsets, (1, BLOCK_N))
    gathered = ct.gather(table_ptr, (rows_2d, cols_2d))
    ct.store(emb_out_ptr, index=(tok, col), tile=gathered)


@oracle_impl(hardware="B200", point="174c0900", BLOCK_M=1, BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    ids, table = inputs
    add_out = torch.empty_strided(
        (BATCH, SEQ_LEN),
        (SEQ_LEN, 1),
        device=ids.device,
        dtype=torch.int64,
    )
    emb_out = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=torch.float32,
    )

    ids_1d = ids.view(-1)
    add_1d = add_out.view(-1)
    emb_2d = emb_out.view(TOKENS, HIDDEN)

    # Tile columns with BLOCK_N=256 (HIDDEN=768 = 256*3)
    COL_BLOCK = 256
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(TOKENS, BLOCK_M), HIDDEN // COL_BLOCK, 1),
        _embedding_shifted_gather_kernel,
        (ids_1d, table, add_1d, emb_2d, BLOCK_M, COL_BLOCK),
    )
    return add_out, emb_out
