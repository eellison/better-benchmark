"""cuTile port of pointwise_36787f0bfc90: MobileBERT shifted-neighbor embedding cat.

BANDWIDTH_BOUND: the output [32768, 384] concatenates three per-row bf16
[128] embedding tiles: next-token, current-token, prev-token (with zero fill
at seq boundaries).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 128
OUT_WIDTH = 384


@ct.kernel
def _cat_three_kernel(
    next_ptr,     # bf16 [rows, HIDDEN]
    current_ptr,  # bf16 [rows, HIDDEN]
    prev_ptr,     # bf16 [rows, HIDDEN]
    out_ptr,      # bf16 [rows, OUT_WIDTH]
    OUT_WIDTH_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    row_block = ct.bid(0)
    nxt = ct.load(next_ptr, index=(row_block, 0), shape=(BLOCK_M, HIDDEN_))
    cur = ct.load(current_ptr, index=(row_block, 0), shape=(BLOCK_M, HIDDEN_))
    prv = ct.load(prev_ptr, index=(row_block, 0), shape=(BLOCK_M, HIDDEN_))
    ct.store(out_ptr, index=(row_block, 0, 0), tile=ct.reshape(nxt, (BLOCK_M, 1, HIDDEN_)))
    ct.store(out_ptr, index=(row_block, 1, 0), tile=ct.reshape(cur, (BLOCK_M, 1, HIDDEN_)))
    ct.store(out_ptr, index=(row_block, 2, 0), tile=ct.reshape(prv, (BLOCK_M, 1, HIDDEN_)))


@oracle_impl(hardware="B200", point="875862cf", BLOCK_M=16)
def oracle_forward(inputs, *, BLOCK_M: int):
    table, ids, _shape_param_0 = inputs
    rows = 32768
    assert rows % BLOCK_M == 0

    # ids has shape [256, 128]: batches x seq_len
    batch_size, seq_len = 256, 128
    ids_bs = ids.view(batch_size, seq_len)

    # Compute the three per-row token id vectors (with zeros at boundaries).
    zero_col = torch.zeros((batch_size, 1), dtype=torch.int64, device=ids.device)
    next_ids = torch.cat([ids_bs[:, 1:], zero_col], dim=1)
    prev_ids = torch.cat([zero_col, ids_bs[:, :-1]], dim=1)
    positions = torch.arange(seq_len, device=ids.device)
    next_valid = positions < (seq_len - 1)
    prev_valid = positions > 0
    next_flat = next_ids.view(rows)
    current_flat = ids_bs.view(rows)
    prev_flat = prev_ids.view(rows)

    device = table.device
    next_emb = torch.embedding(table, next_flat)
    current_emb = torch.embedding(table, current_flat)
    prev_emb = torch.embedding(table, prev_flat)

    # Zero the boundary rows (broadcast (seq_len,) mask across batches).
    # After torch.embedding, next_emb is (rows, HIDDEN) contiguous.
    next_mask_2d = next_valid.view(1, seq_len, 1).expand(batch_size, seq_len, HIDDEN)
    prev_mask_2d = prev_valid.view(1, seq_len, 1).expand(batch_size, seq_len, HIDDEN)
    next_emb_2d = next_emb.view(batch_size, seq_len, HIDDEN)
    prev_emb_2d = prev_emb.view(batch_size, seq_len, HIDDEN)
    next_emb_2d = next_emb_2d * next_mask_2d
    prev_emb_2d = prev_emb_2d * prev_mask_2d
    # next_emb_2d and prev_emb_2d are fresh from the mul so already contiguous
    # — use reshape (metadata-only) instead of contiguous() + view.
    next_emb = next_emb_2d.reshape(rows, HIDDEN)
    prev_emb = prev_emb_2d.reshape(rows, HIDDEN)

    out = torch.empty_strided(
        (rows, OUT_WIDTH),
        (OUT_WIDTH, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_3d = out.view(rows, 3, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows // BLOCK_M, 1, 1),
        _cat_three_kernel,
        (next_emb, current_emb, prev_emb, out_3d, OUT_WIDTH, HIDDEN, BLOCK_M),
    )
    return out
