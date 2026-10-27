"""cuTile port of var_mean_ef83fb4a9dbc: FNet embedding assembly + LayerNorm.

Fuses word/token-type/position embedding gathers, the two f32 adds, and
LayerNorm into one kernel — matches the Triton structure. Also emits the
side outputs (normalized, affine, invstd/HIDDEN) directly from the kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
EPS = 1.0e-12


@ct.kernel
def _fnet_embedding_layernorm_kernel(
    token_type_ids_ptr,   # i64 [SEQ_LEN]
    word_table_ptr,       # f32 [VOCAB, HIDDEN]
    token_ids_ptr,        # i64 [ROWS]
    token_type_table_ptr, # f32 [4, HIDDEN]
    position_table_ptr,   # f32 [SEQ_LEN, HIDDEN]
    position_ids_ptr,     # i64 [SEQ_LEN]
    weight_ptr,           # f32 [HIDDEN]
    bias_ptr,             # f32 [HIDDEN]
    normalized_ptr,       # f32 [ROWS, HIDDEN]
    affine_ptr,           # f32 [ROWS, HIDDEN]
    div_ptr,              # f32 [ROWS]
    SEQ_LEN: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    seq = row % SEQ_LEN

    ttype_id_tile = ct.load(token_type_ids_ptr, index=(seq,), shape=(1,))
    token_id_tile = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    pos_id_tile = ct.load(position_ids_ptr, index=(seq,), shape=(1,))
    ttype_id = ct.reshape(ttype_id_tile, ())
    token_id = ct.reshape(token_id_tile, ())
    pos_id = ct.reshape(pos_id_tile, ())

    word = ct.load(word_table_ptr, index=(token_id, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    ttype = ct.load(token_type_table_ptr, index=(ttype_id, 0),
                    shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO)
    position = ct.load(position_table_ptr, index=(pos_id, 0),
                       shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO)

    add_0 = word + ttype
    x = add_0 + position

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(cols < HIDDEN_C, (1, BLOCK_H))
    zero_2d = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    x_masked = ct.where(col_mask, x, zero_2d)
    inv_h = 1.0 / HIDDEN_C
    mean = ct.sum(x_masked) * inv_h
    centered = ct.where(col_mask, x - mean, zero_2d)
    variance = ct.sum(centered * centered) * inv_h
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(div_ptr, index=(row,),
             tile=ct.reshape(invstd * inv_h, (1,)))


@oracle_impl(hardware="B200", point="9a73f5a0", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        token_type_ids,   # i64 [1, 512]
        word_table,       # f32 [32000, 768]
        token_ids,        # i64 [32, 512]
        token_type_table, # f32 [4, 768]
        position_table,   # f32 [512, 768]
        position_ids,     # i64 [1, 512]
        weight,           # f32 [768]
        bias,             # f32 [768]
        _expand_shape,
        flat_shape,
    ) = inputs
    device = word_table.device
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(weight.shape[0])
    rows = batch * seq_len

    # Padded scratch for normalized and affine outputs; we'll copy the valid
    # region back to the strided returns.
    norm_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    token_type_ids_1d = token_type_ids.view(-1)
    token_ids_1d = token_ids.view(-1)
    position_ids_1d = position_ids.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _fnet_embedding_layernorm_kernel,
        (token_type_ids_1d, word_table, token_ids_1d, token_type_table,
         position_table, position_ids_1d, weight, bias,
         norm_pad, affine_pad, div_1d,
         seq_len, hidden, BLOCK_H),
    )

    normalized = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=device, dtype=torch.float32,
    )
    normalized.view(rows, hidden).copy_(norm_pad.narrow(1, 0, hidden))

    flat_shape_t = tuple(int(x) for x in flat_shape)
    affine = torch.empty_strided(
        flat_shape_t, (hidden, 1),
        device=device, dtype=torch.float32,
    )
    affine.view(rows, hidden).copy_(affine_pad.narrow(1, 0, hidden))

    div = torch.empty_strided(
        (batch, seq_len, 1), (seq_len, 1, 1),
        device=device, dtype=torch.float32,
    )
    div.view(rows).copy_(div_1d)

    return normalized, affine, div
