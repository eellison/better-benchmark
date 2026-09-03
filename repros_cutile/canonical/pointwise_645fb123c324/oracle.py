"""cuTile port of pointwise_645fb123c324: TrOCR generated-position embedding.

Two outputs:
1. indices: i64 [batch, SEQ_LEN=256] where indices[b, t] = t + 2 (broadcast across batch).
2. embedding: f32 [batch, SEQ_LEN, HIDDEN] where embedding[b, t, :] = table[t+2, :].

Table is f32[514, 1024]. We separately fill indices and embedding.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ_LEN = 256
ROW_OFFSET = 2


@ct.kernel
def _embedding_kernel(
    table_ptr,      # f32 [table_rows, HIDDEN]
    embedding_ptr,  # f32 [batch, SEQ_LEN, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    token = ct.bid(0)
    col_block = ct.bid(1)
    batch = ct.bid(2)
    row = token + ROW_OFFSET
    values = ct.load(table_ptr, index=(row, col_block), shape=(1, BLOCK_D))
    # Store into [batch, token, col_block*BLOCK_D:(col_block+1)*BLOCK_D]
    tile = ct.reshape(values, (1, 1, BLOCK_D))
    ct.store(embedding_ptr, index=(batch, token, col_block), tile=tile)


@ct.kernel
def _indices_kernel(
    indices_ptr,  # i64 [batch, SEQ_LEN]
    BLOCK_T: ct.Constant[int],
):
    batch = ct.bid(0)
    t_block = ct.bid(1)
    t = ct.arange(BLOCK_T, dtype=ct.int64)
    row = t + t_block * BLOCK_T + ROW_OFFSET
    row_2d = ct.reshape(row, (1, BLOCK_T))
    ct.store(indices_ptr, index=(batch, t_block), tile=row_2d)


@oracle_impl(hardware="B200", point="59aa3e3a", BLOCK_D=1024, BLOCK_T=SEQ_LEN)
def oracle_forward(inputs, *, BLOCK_D: int, BLOCK_T: int):
    table, shape_param = inputs
    batch = int(shape_param[0])
    hidden = int(table.shape[1])

    indices = torch.empty_strided(
        (batch, SEQ_LEN), (SEQ_LEN, 1),
        device=table.device, dtype=torch.int64,
    )
    embedding = torch.empty_strided(
        (batch, SEQ_LEN, hidden), (SEQ_LEN * hidden, hidden, 1),
        device=table.device, dtype=torch.float32,
    )

    # BLOCK_D=1024 = hidden -> just 1 tile per row.
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (SEQ_LEN, hidden // BLOCK_D, batch),
        _embedding_kernel, (table, embedding, hidden, BLOCK_D),
    )
    ct.launch(
        stream, (batch, SEQ_LEN // BLOCK_T, 1),
        _indices_kernel, (indices, BLOCK_T),
    )
    return indices, embedding
