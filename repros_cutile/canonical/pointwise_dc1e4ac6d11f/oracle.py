"""cuTile port of pointwise_dc1e4ac6d11f: embedding gather with +2 shifted index side output."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _embedding_add_kernel(
    ids_ptr,      # (SEQ_LEN,) i64
    table_ptr,    # (VOCAB, HIDDEN) f32
    add_out_ptr,  # (SEQ_LEN,) i64
    emb_out_ptr,  # (SEQ_LEN, HIDDEN) f32
    HIDDEN: ct.Constant[int],
    HIDDEN_PAD: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    tok_tile = ct.bid(0)
    ids = ct.load(ids_ptr, index=(tok_tile,), shape=(BLOCK_M,))
    shifted = ids + ct.full(shape=(BLOCK_M,), fill_value=2, dtype=ct.int64)
    ct.store(add_out_ptr, index=(tok_tile,), tile=shifted)

    # 2D gather: for each token, gather its full row from table.
    # indices tuple: (row_indices[BLOCK_M, 1], col_indices[1, HIDDEN_PAD])
    row_idx = ct.reshape(shifted, (BLOCK_M, 1))
    col_arange = ct.arange(HIDDEN_PAD, dtype=ct.int64)
    col_idx = ct.reshape(col_arange, (1, HIDDEN_PAD))
    # Mask cols to valid range (padding lanes will be masked out on store)
    col_mask_1d = col_arange < HIDDEN
    col_mask = ct.reshape(col_mask_1d, (1, HIDDEN_PAD))
    # For OOB cols, redirect to col 0 (safe read); result masked out on store
    safe_col_idx = ct.where(col_mask, col_idx, ct.reshape(
        ct.full(shape=(HIDDEN_PAD,), fill_value=0, dtype=ct.int64),
        (1, HIDDEN_PAD),
    ))
    values = ct.gather(table_ptr, (row_idx, safe_col_idx))

    # Scatter valid values to (BLOCK_M, HIDDEN) region
    tok_arange = ct.reshape(tok_tile * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int64), (BLOCK_M, 1))
    flat_idx = tok_arange * HIDDEN + col_idx
    ct.scatter(emb_out_ptr, flat_idx, values, mask=col_mask)


@oracle_impl(hardware="B200", point="ca400fad", HIDDEN=1024, HIDDEN_PAD=1024, BLOCK_M=1)
@oracle_impl(hardware="B200", point="df436a45", HIDDEN=768, HIDDEN_PAD=1024, BLOCK_M=1)
def oracle_forward(inputs, *, HIDDEN, HIDDEN_PAD, BLOCK_M):
    ids, table = inputs
    add_out = torch.empty_strided(
        (1, 1024),
        (1024, 1),
        device=ids.device,
        dtype=torch.int64,
    )
    emb_out = torch.empty_strided(
        (1, 1024, HIDDEN),
        (1024 * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=torch.float32,
    )

    # View outputs as 2D (rank 1) tensors for cuTile compatibility.
    add_2d = add_out.view(1024)
    emb_flat = emb_out.view(1024 * HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1024 // BLOCK_M, 1, 1),
        _embedding_add_kernel,
        (ids, table, add_2d, emb_flat, HIDDEN, HIDDEN_PAD, BLOCK_M),
    )
    return add_out, emb_out
