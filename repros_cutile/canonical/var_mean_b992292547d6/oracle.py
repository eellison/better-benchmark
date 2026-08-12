"""cuTile port of var_mean_b992292547d6: FNet embedding assembly + LayerNorm.

Per row (batch*seq_len): sum(word_embed + token_type_embed + position_embed);
then population var_mean + affine LayerNorm with eps=1e-12.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _fnet_embedding_layernorm_kernel(
    word_table_ptr,        # f32 [vocab, HIDDEN]
    token_ids_ptr,         # i64 [batch, seq_len]
    token_type_ids_ptr,    # i64 [1, seq_len]
    token_type_table_ptr,  # f32 [types, HIDDEN]
    position_table_ptr,    # f32 [seq_len, HIDDEN]
    position_ids_ptr,      # i64 [1, seq_len]
    weight_ptr,            # f32 [HIDDEN]
    bias_ptr,              # f32 [HIDDEN]
    out_ptr,               # f32 [rows, HIDDEN]
    SEQ_LEN: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    seq_idx = row - (row // SEQ_LEN) * SEQ_LEN  # row % SEQ_LEN

    # Load ids as scalars (shape=(1,))
    token_id = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    token_type_id = ct.load(token_type_ids_ptr, index=(seq_idx,), shape=(1,))
    position_id = ct.load(position_ids_ptr, index=(seq_idx,), shape=(1,))

    # Gather embeddings: word_table[token_id, :]
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    valid_1d = cols < HIDDEN
    # token_id is a tile (1,) int64; broadcast against cols (BLOCK_H,)
    tid_bcast = ct.reshape(ct.astype(token_id, ct.int32), (1,)) + ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    ttid_bcast = ct.reshape(ct.astype(token_type_id, ct.int32), (1,)) + ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    pid_bcast = ct.reshape(ct.astype(position_id, ct.int32), (1,)) + ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    zero_col = ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    cols_masked = ct.where(valid_1d, cols, zero_col)
    word = ct.gather(word_table_ptr, (tid_bcast, cols_masked), mask=valid_1d, padding_value=0.0)
    token_type = ct.gather(token_type_table_ptr, (ttid_bcast, cols_masked), mask=valid_1d, padding_value=0.0)
    position = ct.gather(position_table_ptr, (pid_bcast, cols_masked), mask=valid_1d, padding_value=0.0)

    x = word + token_type + position
    valid = valid_1d
    zero_f = ct.full(shape=(BLOCK_H,), fill_value=0.0, dtype=ct.float32)
    x_masked = ct.where(valid, x, zero_f)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(valid, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    affine = normalized * weight + bias

    r_idx = ct.full(shape=(BLOCK_H,), fill_value=row, dtype=ct.int32)
    ct.scatter(out_ptr, (r_idx, cols), ct.where(valid, affine, zero_f), mask=valid)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="9a73f5a0", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    (
        word_table,
        token_ids,
        token_type_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        _expand_shape,
        flat_shape,
    ) = inputs

    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(weight.shape[0])
    rows = batch * seq_len
    flat_shape = _as_shape(flat_shape)

    out = torch.empty_strided(
        flat_shape,
        (hidden, 1),
        device=word_table.device,
        dtype=torch.float32,
    )
    # Flatten ids to 1D
    token_ids_1d = token_ids.view(rows)
    token_type_ids_1d = token_type_ids.view(seq_len)
    position_ids_1d = position_ids.view(seq_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _fnet_embedding_layernorm_kernel,
        (word_table, token_ids_1d, token_type_ids_1d, token_type_table,
         position_table, position_ids_1d, weight, bias, out,
         seq_len, hidden, BLOCK_H),
    )
    return out
