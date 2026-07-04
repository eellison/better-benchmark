"""cuTile port of var_mean_1e8c0ecafa26: YituTechConvBert embedding + LayerNorm.

For each row in [batch*seq_len]: gather word/position/token_type embeddings, sum with bf16
round-trip semantics, LayerNorm eps=1e-12, affine bf16 out.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _embedding_layernorm_kernel(
    word_table_ptr,        # bf16 [vocab, HIDDEN]
    word_ids_ptr,          # i64 [batch, seq_len]
    position_table_ptr,    # bf16 [seq_len, HIDDEN]
    position_ids_ptr,      # i64 [1, seq_len]
    token_type_ids_ptr,    # i64 [1, seq_len]
    token_type_table_ptr,  # bf16 [types, HIDDEN]
    weight_ptr,            # bf16 [HIDDEN]
    bias_ptr,              # bf16 [HIDDEN]
    out_ptr,               # bf16 [rows, HIDDEN]
    SEQ_LEN: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    seq_idx = row - (row // SEQ_LEN) * SEQ_LEN  # row % SEQ_LEN

    word_id = ct.load(word_ids_ptr, index=(row,), shape=(1,))
    token_type_id = ct.load(token_type_ids_ptr, index=(seq_idx,), shape=(1,))
    position_id = ct.load(position_ids_ptr, index=(seq_idx,), shape=(1,))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    valid_1d = cols < HIDDEN

    tid_bcast = ct.reshape(ct.astype(word_id, ct.int32), (1,)) + ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    pid_bcast = ct.reshape(ct.astype(position_id, ct.int32), (1,)) + ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    ttid_bcast = ct.reshape(ct.astype(token_type_id, ct.int32), (1,)) + ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    zero_col = ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    cols_masked = ct.where(valid_1d, cols, zero_col)
    word_bf = ct.gather(word_table_ptr, (tid_bcast, cols_masked), mask=valid_1d, padding_value=0)
    position_bf = ct.gather(position_table_ptr, (pid_bcast, cols_masked), mask=valid_1d, padding_value=0)
    token_type_bf = ct.gather(token_type_table_ptr, (ttid_bcast, cols_masked), mask=valid_1d, padding_value=0)
    word = ct.astype(word_bf, ct.float32)
    position = ct.astype(position_bf, ct.float32)
    token_type = ct.astype(token_type_bf, ct.float32)

    # Non-inductor (eager-compatible) numerics with bf16 round-trip
    add0 = ct.astype(ct.astype(word + position, ct.bfloat16), ct.float32)
    x = ct.astype(ct.astype(add0 + token_type, ct.bfloat16), ct.float32)

    zero_f = ct.full(shape=(BLOCK_H,), fill_value=0.0, dtype=ct.float32)
    x_masked = ct.where(valid_1d, x, zero_f)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(valid_1d, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                               padding_mode=ct.PaddingMode.ZERO), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                             padding_mode=ct.PaddingMode.ZERO), ct.float32)
    affine = normalized * weight + bias
    out_bf = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.full(shape=(BLOCK_H,), fill_value=0.0, dtype=ct.bfloat16)

    r_idx = ct.full(shape=(BLOCK_H,), fill_value=row, dtype=ct.int32)
    ct.scatter(out_ptr, (r_idx, cols),
               ct.where(valid_1d, out_bf, zero_bf), mask=valid_1d)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="a6271911", BLOCK_M=2, BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_H):
    (
        word_table,
        word_ids,
        position_table,
        position_ids,
        token_type_ids,
        token_type_table,
        weight,
        bias,
        _expand_shape,
        view_shape0,
        view_shape1,
        view_shape2,
        view_shape3,
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
    word_ids_1d = word_ids.view(rows)
    position_ids_1d = position_ids.view(seq_len)
    token_type_ids_1d = token_type_ids.view(seq_len)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _embedding_layernorm_kernel,
        (word_table, word_ids_1d, position_table, position_ids_1d, token_type_ids_1d,
         token_type_table, weight, bias, out_2d,
         seq_len, hidden, BLOCK_H),
    )
    return (
        out,
        out.view(_as_shape(view_shape0)),
        out.view(_as_shape(view_shape1)),
        out.view(_as_shape(view_shape2)),
        out.view(_as_shape(view_shape3)),
        out.permute(0, 2, 1),
    )
