"""cuTile port of var_mean_d1f8c258a72a: RoBERTa embedding gather + LayerNorm.

Ports the Triton `_roberta_embedding_layernorm_kernel` — gather word,
token_type, and position embeddings (with computed indices), then bf16-
boundary add, then LayerNorm across HIDDEN=768 with weight/bias affine.

HIDDEN=768 is not a power of 2; tile size is 1024 with mask-based scatter to
write only valid columns back to the [rows, HIDDEN] output (no padded copy).
Mirrors Triton's BLOCK_M=2 row block.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPSILON = 1.0e-12


@ct.kernel
def _roberta_embedding_layernorm_kernel(
    word_table_ptr,           # bf16 [VOCAB, HIDDEN]
    word_ids_ptr,             # i64  [ROWS]
    token_type_source_ptr,    # i64  [SEQ_LEN]
    position_index_source_ptr,# i64  [ROWS]
    position_mask_ptr,        # i32  [ROWS]
    token_type_table_ptr,     # bf16 [TT_VOCAB, HIDDEN]
    position_table_ptr,       # bf16 [POS_VOCAB, HIDDEN]
    weight_ptr,               # bf16 [HIDDEN]
    bias_ptr,                 # bf16 [HIDDEN]
    out_flat_ptr,             # bf16 [ROWS * HIDDEN] (flat)
    ROWS: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    EPS_C: ct.Constant[float],
    BLOCK_M: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row_block = ct.bid(0)
    rows_1d = ct.arange(BLOCK_M, dtype=ct.int64) + row_block * BLOCK_M
    cols_1d = ct.arange(BLOCK_H, dtype=ct.int64)

    row_mask_1d = rows_1d < ROWS
    col_mask_1d = cols_1d < HIDDEN_C
    rows_2d = ct.reshape(rows_1d, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols_1d, (1, BLOCK_H))
    row_mask_2d = ct.reshape(row_mask_1d, (BLOCK_M, 1))
    col_mask_2d = ct.reshape(col_mask_1d, (1, BLOCK_H))
    mask_2d = row_mask_2d & col_mask_2d

    # Safe row for gathers when out-of-range
    zero_1d = ct.zeros((BLOCK_M,), dtype=ct.int64)
    safe_rows_1d = ct.where(row_mask_1d, rows_1d, zero_1d)

    word_id = ct.gather(word_ids_ptr, (safe_rows_1d,))
    raw_index = ct.gather(position_index_source_ptr, (safe_rows_1d,))
    index_mask = ct.gather(position_mask_ptr, (safe_rows_1d,))
    position_id = raw_index * ct.astype(index_mask, ct.int64)
    token_type_id = ct.gather(token_type_source_ptr, (position_id,))

    word_id_2d = ct.reshape(word_id, (BLOCK_M, 1))
    tt_id_2d = ct.reshape(token_type_id, (BLOCK_M, 1))
    pos_id_2d = ct.reshape(position_id, (BLOCK_M, 1))

    # Broadcast via multiply by ones
    ones_bh = ct.full((BLOCK_M, BLOCK_H), 1, dtype=ct.int64)
    word_row_2d = word_id_2d * ones_bh
    tt_row_2d = tt_id_2d * ones_bh
    pos_row_2d = pos_id_2d * ones_bh
    zero_cols = ct.zeros((BLOCK_M, BLOCK_H), dtype=ct.int64)
    col_bcast = cols_2d * ones_bh
    safe_cols = ct.where(col_mask_2d, col_bcast, zero_cols)

    word = ct.gather(word_table_ptr, (word_row_2d, safe_cols))
    token_type = ct.gather(token_type_table_ptr, (tt_row_2d, safe_cols))
    position = ct.gather(position_table_ptr, (pos_row_2d, safe_cols))

    word_f = ct.astype(word, ct.float32)
    token_type_f = ct.astype(token_type, ct.float32)
    position_f = ct.astype(position, ct.float32)

    add0 = ct.astype(ct.astype(word_f + token_type_f, ct.bfloat16), ct.float32)
    x = ct.astype(ct.astype(add0 + position_f, ct.bfloat16), ct.float32)

    zero_bh = ct.zeros((BLOCK_M, BLOCK_H), dtype=ct.float32)
    x_reduced = ct.where(mask_2d, x, zero_bh)

    inv_h = 1.0 / float(HIDDEN_C)
    mean = ct.sum(x_reduced, axis=1, keepdims=True) * inv_h
    centered = x - mean
    centered_masked = ct.where(mask_2d, centered, zero_bh)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS_C)
    normalized = centered * invstd

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                        padding_mode=ct.PaddingMode.ZERO)
    bias_1d = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                      padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(ct.reshape(weight_1d, (1, BLOCK_H)), ct.float32)
    bias_f = ct.astype(ct.reshape(bias_1d, (1, BLOCK_H)), ct.float32)
    affine = normalized * weight_f + bias_f

    # Scatter into flat (ROWS*HIDDEN) — write only valid (row, col) elements.
    out_offsets = rows_2d * HIDDEN_C + cols_2d * ct.full((BLOCK_M, 1), 1, dtype=ct.int64)
    ct.scatter(out_flat_ptr, out_offsets, ct.astype(affine, ct.bfloat16), mask=mask_2d)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="975c5fbc", BLOCK_M=2, BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    (
        word_table,
        word_ids,
        token_type_source,
        position_index_source,
        position_mask,
        token_type_table,
        position_table,
        weight,
        bias,
        _shape0,
        _shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs
    batch = int(word_ids.shape[0])
    seq_len = int(word_ids.shape[1])
    hidden = int(word_table.shape[1])
    rows = batch * seq_len
    device = word_table.device

    word_ids_1d = word_ids.reshape(rows)
    position_index_source_1d = position_index_source.reshape(rows)
    position_mask_1d = position_mask.reshape(rows)
    tt_source_1d = token_type_source.reshape(-1)

    out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_flat = out.view(rows * hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), 1, 1),
        _roberta_embedding_layernorm_kernel,
        (word_table, word_ids_1d, tt_source_1d, position_index_source_1d,
         position_mask_1d, token_type_table, position_table,
         weight, bias, out_flat, rows, hidden, EPSILON, BLOCK_M, BLOCK_H),
    )

    return (
        out,
        out.view(_shape_tuple(shape2)),
        out.view(_shape_tuple(shape3)),
        out.view(_shape_tuple(shape4)),
    )
