"""cuTile port of var_mean_14955ed02f87: BERT embedding + LayerNorm inference.

Fuses word/token-type/position embedding gathers, two bf16 add-round steps,
and LayerNorm into one kernel — matches the Triton structure.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _embedding_layernorm_kernel(
    word_table_ptr,          # bf16 [30522, HIDDEN]
    word_ids_ptr,            # i64 [ROWS]
    token_type_source_ptr,   # i64 [SEQ_LEN]
    position_ids_ptr,        # i64 [SEQ_LEN]
    token_type_table_ptr,    # bf16 [2, HIDDEN]
    position_table_ptr,      # bf16 [SEQ_LEN, HIDDEN]
    weight_ptr,              # bf16 [HIDDEN]
    bias_ptr,                # bf16 [HIDDEN]
    out_ptr,                 # bf16 [ROWS, HIDDEN]
    SEQ_LEN: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    seq = row % SEQ_LEN
    # Load ids as scalars via 1-elt tiles.
    word_id_tile = ct.load(word_ids_ptr, index=(row,), shape=(1,))
    position_id_tile = ct.load(position_ids_ptr, index=(seq,), shape=(1,))
    position_id = ct.reshape(position_id_tile, ())
    token_type_id_tile = ct.load(
        token_type_source_ptr, index=(position_id,), shape=(1,))
    token_type_id = ct.reshape(token_type_id_tile, ())
    word_id = ct.reshape(word_id_tile, ())

    word = ct.load(word_table_ptr, index=(word_id, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    token_type = ct.load(token_type_table_ptr, index=(token_type_id, 0),
                         shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO)
    position = ct.load(position_table_ptr, index=(position_id, 0),
                       shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO)

    add0 = ct.astype(
        ct.astype(word, ct.float32) + ct.astype(token_type, ct.float32),
        ct.bfloat16)
    x_bf = ct.astype(
        ct.astype(add0, ct.float32) + ct.astype(position, ct.float32),
        ct.bfloat16)
    x = ct.astype(x_bf, ct.float32)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    valid = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    zero_2d = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    x_masked = ct.where(valid, x, zero_2d)

    inv_h = 1.0 / HIDDEN
    mean = ct.sum(x_masked, axis=1, keepdims=True) * inv_h
    centered = ct.where(valid, x - mean, zero_2d)
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                padding_mode=ct.PaddingMode.ZERO), ct.float32)
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                padding_mode=ct.PaddingMode.ZERO), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    y = centered * invstd * weight_2d + bias_2d
    # Use scatter to write only in-bounds columns for out.
    r_idx = ct.full(shape=(1, BLOCK_H), fill_value=row, dtype=ct.int32)
    c_idx = ct.reshape(cols, (1, BLOCK_H))
    ct.scatter(out_ptr, (r_idx, c_idx),
               ct.astype(y, ct.bfloat16), mask=valid)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="a6271911", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        word_table,
        word_ids,
        token_type_source,
        position_ids,
        token_type_table,
        position_table,
        weight,
        bias,
        _token_type_shape,
        view_shape0,
        view_shape1,
        view_shape2,
    ) = inputs

    batch = int(word_ids.shape[0])
    seq_len = int(word_ids.shape[1])
    hidden = int(word_table.shape[1])
    rows = batch * seq_len

    out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )
    out_2d = out.view(rows, hidden)

    word_ids_flat = word_ids.view(-1)
    position_ids_flat = position_ids.view(-1)
    token_type_source_flat = token_type_source.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _embedding_layernorm_kernel,
        (word_table, word_ids_flat, token_type_source_flat, position_ids_flat,
         token_type_table, position_table, weight, bias, out_2d,
         seq_len, hidden, BLOCK_H),
    )

    return (
        out,
        out.view(_as_shape(view_shape0)),
        out.view(_as_shape(view_shape1)),
        out.view(_as_shape(view_shape2)),
    )
