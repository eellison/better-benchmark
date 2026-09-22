"""cuTile port of var_mean_a73bc583f18c: DeBERTa embedding+LayerNorm inference.

Ports the Triton `_deberta_embedding_layernorm_kernel` — gather word and
position embeddings, cast-to-bf16 add-then-cast-to-fp32 boundary, LayerNorm
across HIDDEN=1536 with weight/bias affine, store bf16.

HIDDEN=1536 is not a power of 2; the tile size is 2048. We store into a
padded [ROWS, 2048] buffer and slice down to (ROWS, 1536) for the final
returned tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-7


@ct.kernel
def _deberta_embedding_layernorm_kernel(
    word_table_ptr,       # bf16 [VOCAB, HIDDEN]
    word_ids_ptr,         # i64  [ROWS]
    position_table_ptr,   # bf16 [SEQ_LEN, HIDDEN]
    position_ids_ptr,     # i64  [SEQ_LEN]
    weight_ptr,           # bf16 [HIDDEN]
    bias_ptr,             # bf16 [HIDDEN]
    out_ptr,              # bf16 [ROWS, BLOCK_H] (padded)
    SEQ_LEN: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    EPSILON: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    # Compute seq = row % SEQ_LEN
    seq = row - (row // SEQ_LEN) * SEQ_LEN

    # Gather word_id[row] and position_id[seq]
    word_id_tile = ct.load(word_ids_ptr, index=(row,), shape=(1,))
    position_id_tile = ct.load(position_ids_ptr, index=(seq,), shape=(1,))

    # Load the corresponding embedding rows via gather (2D).
    cols_arange = ct.arange(BLOCK_H, dtype=ct.int64)
    col_mask = cols_arange < HIDDEN
    # Clamp out-of-range cols to 0 (they'll be masked out for reductions/stores).
    zero_cols = ct.zeros((BLOCK_H,), dtype=ct.int64)
    safe_cols = ct.where(col_mask, cols_arange, zero_cols)

    # word_id and position_id are (1,) tiles; reshape for broadcasting
    word_id_2d = ct.reshape(word_id_tile, (1, 1))
    position_id_2d = ct.reshape(position_id_tile, (1, 1))
    cols_2d = ct.reshape(safe_cols, (1, BLOCK_H))

    # Broadcast to (1, BLOCK_H) via + 0 * cols_2d (or direct 2D gather).
    row_indices_word = ct.reshape(
        word_id_tile + ct.zeros((1,), dtype=ct.int64), (1, 1)
    )
    row_indices_pos = ct.reshape(
        position_id_tile + ct.zeros((1,), dtype=ct.int64), (1, 1)
    )

    # For (1, BLOCK_H) tile output, indices need to be (1, BLOCK_H).
    # Broadcast row_indices by mul with ones.
    ones_bh = ct.full((1, BLOCK_H), 1, dtype=ct.int64)
    word_row_bc = row_indices_word * ones_bh
    pos_row_bc = row_indices_pos * ones_bh

    word = ct.gather(word_table_ptr, (word_row_bc, cols_2d))
    position = ct.gather(position_table_ptr, (pos_row_bc, cols_2d))

    word_f = ct.astype(word, ct.float32)
    position_f = ct.astype(position, ct.float32)
    # Casting through bf16 to match rtne boundary; cuTile default is already
    # round-to-nearest for bf16.
    x = ct.astype(ct.astype(word_f + position_f, ct.bfloat16), ct.float32)

    # Zero out OOB columns for reductions.
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))
    zero_bh = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    x_reduced = ct.where(col_mask_2d, x, zero_bh)

    inv_h = 1.0 / float(HIDDEN)
    mean = ct.sum(x_reduced, axis=1, keepdims=True) * inv_h  # (1, 1)
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, zero_bh)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPSILON)
    normalized = centered * invstd

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                        padding_mode=ct.PaddingMode.ZERO)
    bias_1d = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                      padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(ct.reshape(weight_1d, (1, BLOCK_H)), ct.float32)
    bias_f = ct.astype(ct.reshape(bias_1d, (1, BLOCK_H)), ct.float32)
    affine = normalized * weight_f + bias_f
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="20cc819f", BLOCK_M=1, BLOCK_H=2048)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    (
        word_table,
        word_ids,
        position_table,
        position_ids,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
    ) = inputs

    batch = int(word_ids.shape[0])
    seq_len = int(word_ids.shape[1])
    hidden = int(weight.shape[0])
    rows = batch * seq_len
    device = word_table.device

    # word_ids [8, 512] -> flat (4096,)
    word_ids_1d = word_ids.reshape(rows)
    position_ids_1d = position_ids.reshape(seq_len)

    # Padded output
    out_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _deberta_embedding_layernorm_kernel,
        (word_table, word_ids_1d, position_table, position_ids_1d,
         weight, bias, out_padded, seq_len, hidden, EPS, BLOCK_H),
    )

    # Slice and reshape to output
    out_flat = out_padded[:, :hidden].contiguous()
    out = out_flat.view(batch, seq_len, hidden)

    return (
        out,
        out.view(_as_shape(shape0)),
        out.view(_as_shape(shape1)),
        out.view(_as_shape(shape2)),
    )
