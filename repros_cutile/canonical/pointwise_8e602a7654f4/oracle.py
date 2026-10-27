"""cuTile port of pointwise_8e602a7654f4: MobileBERT shifted-neighbor embedding cat.

Gathers next-token, current-token, prev-token rows from an f32 embedding table
into a bf16 [32768, 384] output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 256
SEQ = 128
HIDDEN = 128
ROWS = BATCH * SEQ
OUT_HIDDEN = HIDDEN * 3


@ct.kernel
def _shifted_embedding_cat_kernel(
    table,       # (30522, 128) f32
    ids,         # (256, 128) i64 (viewed flat as (32768,))
    out,         # (32768, 384) bf16
    SEQ_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    OUT_HIDDEN_: ct.Constant[int],
):
    row = ct.bid(0)
    # Compute batch and seq position
    seq_pos = row - (row // SEQ_) * SEQ_  # row mod SEQ

    # Current token: always use current row
    curr_id = ct.load(ids, index=(row,), shape=(1,))
    curr_id_scalar = ct.reshape(curr_id, ())
    # Actually ct doesn't like 0-d; use (1,).
    cols = ct.arange(HIDDEN_, dtype=ct.int64)
    curr_id_2d = ct.reshape(curr_id, (1, 1))  # broadcast to (1, HIDDEN)
    cols_2d = ct.reshape(cols, (1, HIDDEN_))
    # Gather current row
    curr_values = ct.gather(table, (curr_id_2d, cols_2d))
    curr_bf16 = ct.astype(curr_values, ct.bfloat16)
    # Store into cols [HIDDEN..2*HIDDEN) of output
    curr_bf16 = ct.reshape(curr_bf16, (1, HIDDEN_))
    ct.store(out, index=(row, 1), tile=curr_bf16)  # tile-space column 1 (0..HIDDEN-1 is 0, HIDDEN..2H-1 is 1)

    # Next token: if seq_pos < SEQ-1, use row+1 else use row (masked out)
    next_valid = seq_pos < SEQ_ - 1
    next_id_tile = ct.load(ids, index=(row + 1,), shape=(1,),
                           padding_mode=ct.PaddingMode.ZERO)
    # But this is a runtime shift; we can only do integer runtime shifts in index.
    # The load at index=(row+1,) with shape=(1,) works if row+1 is in bounds.
    # For row = ROWS-1, row+1 = ROWS which is OOB - undefined.
    # Use conditional store approach: load with padding_mode=ZERO would help.
    next_id_2d = ct.reshape(next_id_tile, (1, 1))
    next_values = ct.gather(table, (next_id_2d, cols_2d))
    next_values = ct.astype(next_values, ct.float32)
    # If invalid, use 0
    next_mask_scalar = next_valid
    next_bf16_f32 = ct.where(next_mask_scalar,
                             next_values,
                             ct.zeros(shape=(1, HIDDEN_), dtype=ct.float32))
    next_bf16 = ct.astype(next_bf16_f32, ct.bfloat16)
    ct.store(out, index=(row, 0), tile=next_bf16)

    # Prev token: if seq_pos > 0, use row-1 else use row (masked out)
    prev_valid = seq_pos > 0
    # For row=0 we'd load index -1 which is problematic. Compute index safely
    # by clamping via ct.where — but we can't put runtime values in index.
    # Instead use gather with row-1 as an int tile.
    # Actually simpler: since seq_pos=0 => prev_valid=false, mask out; index
    # would be row-1 which for row=0 is -1. cuTile's gather with negative
    # indices returns padding, so use gather with (row-1) as the row_id.
    # Build the row index tile explicitly
    # But we need a row index; ct.bid gives scalar. Convert to tile:
    row_prev_tile = ct.full(shape=(1, 1), fill_value=row - 1, dtype=ct.int64)  # runtime scalar packed into tile
    # This may not work either — ct.full may need constant fill_value.
    # Alternative: load the ids at (row-1,) via ct.load with padding.
    # But indices for ct.load must be non-negative. For row=0 this is bad.
    # Best: load ids[row] and ids[row-1] via gather with runtime index.
    # Let's compute row_idx explicitly
    prev_row_int = row - 1  # scalar
    # ct.load with negative index may fail. Use ct.load with padding zero:
    prev_id_tile = ct.load(ids, index=(prev_row_int,), shape=(1,),
                           padding_mode=ct.PaddingMode.ZERO)
    prev_id_2d = ct.reshape(prev_id_tile, (1, 1))
    prev_values = ct.gather(table, (prev_id_2d, cols_2d))
    prev_values = ct.astype(prev_values, ct.float32)
    prev_bf16_f32 = ct.where(prev_valid,
                             prev_values,
                             ct.zeros(shape=(1, HIDDEN_), dtype=ct.float32))
    prev_bf16 = ct.astype(prev_bf16_f32, ct.bfloat16)
    ct.store(out, index=(row, 2), tile=prev_bf16)


@oracle_impl(hardware="B200", point="b3d88053")
def oracle_forward(inputs):
    table, ids, shape = inputs
    ids_flat = ids.view(-1)  # (32768,)
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape),
        (OUT_HIDDEN, 1),
        device=table.device, dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ROWS, 1, 1), _shifted_embedding_cat_kernel,
              (table, ids_flat, out, SEQ, HIDDEN, OUT_HIDDEN))
    return out
