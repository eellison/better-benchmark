"""cuTile port of pointwise_a888b5ed0623: XGLM shifted (+2) position embedding gather.

Loads position ids, adds 2, gathers rows from a f32 embedding table.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ_LEN = 128
HIDDEN = 1024


@ct.kernel
def _xglm_shifted_embedding_gather_kernel(
    ids,     # [128] i64
    table,   # [2050, 1024] f32
    out,     # [128, 1024] f32
    HIDDEN_: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    tok = ct.bid(0)
    # Load a block of ids: shape (BLOCK_M,)
    id_tile = ct.load(ids, index=(tok,), shape=(BLOCK_M,))
    shifted = id_tile + ct.full(shape=(BLOCK_M,), fill_value=2, dtype=ct.int64)
    # Column indices
    cols = ct.arange(HIDDEN_, dtype=ct.int64)
    # Broadcast: rows = shifted[:, None], cols = cols[None, :]
    rows_2d = ct.reshape(shifted, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, HIDDEN_))
    values = ct.gather(table, (rows_2d, cols_2d))
    ct.store(out, index=(tok, 0), tile=values)


@oracle_impl(hardware="B200", point="56e670a4", BLOCK_M=128)
def oracle_forward(inputs, *, BLOCK_M):
    ids, table, _shape = inputs
    ids_1d = ids.view(-1)  # [128]
    out = torch.empty_strided(
        (1, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=table.device, dtype=table.dtype,
    )
    out_2d = out.view(SEQ_LEN, HIDDEN)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(SEQ_LEN, BLOCK_M), 1, 1),
              _xglm_shifted_embedding_gather_kernel,
              (ids_1d, table, out_2d, HIDDEN, BLOCK_M))
    return out
