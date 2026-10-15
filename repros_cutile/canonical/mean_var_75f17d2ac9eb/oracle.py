"""cuTile port of mean_var_75f17d2ac9eb: BERT embedding LayerNorm with 3 alias views.

Fuses the word/position/token-type embedding gathers and LayerNorm into
one kernel matching the Triton structure.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _embedding_layernorm_alias_kernel(
    word_table_ptr,        # bf16 [VOCAB, HIDDEN]
    input_ids_ptr,         # i64 [ROWS]
    position_table_ptr,    # bf16 [SEQ, HIDDEN]
    token_type_table_ptr,  # bf16 [2, HIDDEN]
    token_type_ids_ptr,    # i64 [ROWS]
    weight_ptr,            # bf16 [HIDDEN]
    bias_ptr,              # bf16 [HIDDEN]
    add_out_ptr,           # bf16 [ROWS, HIDDEN]
    norm_out_ptr,          # bf16 [ROWS, HIDDEN]
    SEQ: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    seq_index = row % SEQ

    word_id_tile = ct.load(input_ids_ptr, index=(row,), shape=(1,))
    ttype_id_tile = ct.load(token_type_ids_ptr, index=(row,), shape=(1,))
    word_id = ct.reshape(word_id_tile, ())
    ttype_id = ct.reshape(ttype_id_tile, ())

    word = ct.load(word_table_ptr, index=(word_id, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    position = ct.load(position_table_ptr, index=(seq_index, 0),
                       shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO)
    ttype = ct.load(token_type_table_ptr, index=(ttype_id, 0),
                    shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO)

    word_f = ct.astype(word, ct.float32)
    pos_f = ct.astype(position, ct.float32)
    ttype_f = ct.astype(ttype, ct.float32)

    add0 = ct.astype(ct.astype(word_f + pos_f, ct.bfloat16), ct.float32)
    add1 = ct.astype(ct.astype(add0 + ttype_f, ct.bfloat16), ct.float32)
    ct.store(add_out_ptr, index=(row, 0), tile=ct.astype(add1, ct.bfloat16))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    zero_2d = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    add_for_reduce = ct.where(col_mask, add1, zero_2d)

    sum_x = ct.sum(add_for_reduce)
    mean_f32 = sum_x * (1.0 / HIDDEN)
    mean_bf16_f = ct.astype(ct.astype(mean_f32, ct.bfloat16), ct.float32)

    centered_bf16_f = ct.astype(
        ct.astype(add1 - mean_bf16_f, ct.bfloat16), ct.float32
    )
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    scaled_bf16_f = ct.astype(
        ct.astype(weight_2d * centered_bf16_f, ct.bfloat16), ct.float32
    )

    sum_x2 = ct.sum(ct.where(col_mask, add1 * add1, zero_2d))
    variance = (sum_x2 - sum_x * mean_f32) * (1.0 / (HIDDEN - 1.0))
    denom = ct.astype(
        ct.astype(
            ct.sqrt(ct.where(variance > 0.0, variance,
                            ct.astype(0.0, ct.float32))),
            ct.bfloat16),
        ct.float32,
    )
    denom_eps_bf16_f = ct.astype(
        ct.astype(denom + 1.0e-6, ct.bfloat16), ct.float32)

    divided_bf16_f = ct.astype(
        ct.astype(scaled_bf16_f / denom_eps_bf16_f, ct.bfloat16), ct.float32
    )
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    bias_f = ct.astype(bias, ct.float32)
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    out = ct.astype(divided_bf16_f + bias_2d, ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="a655df0f", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        word_table,
        input_ids,
        position_table,
        token_type_table,
        token_type_ids,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
    ) = inputs
    del shape1, shape2

    out_shape = tuple(int(dim) for dim in shape0)
    batch = int(input_ids.shape[0])
    seq = int(input_ids.shape[1])
    hidden = int(weight.shape[0])
    rows = batch * seq

    add_out = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        out_shape, (hidden, 1),
        device=word_table.device, dtype=torch.bfloat16,
    )

    add_out_2d = add_out.view(rows, hidden)
    norm_out_2d = norm_out.view(rows, hidden)
    input_ids_1d = input_ids.view(-1)
    token_type_ids_1d = token_type_ids.view(-1)
    # position_table is (1, seq, hidden) or similar; reshape to (seq, hidden).
    position_table_2d = position_table.view(-1, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _embedding_layernorm_alias_kernel,
        (word_table, input_ids_1d, position_table_2d, token_type_table,
         token_type_ids_1d, weight, bias, add_out_2d, norm_out_2d,
         seq, hidden, BLOCK_H),
    )
    return add_out, norm_out, norm_out, norm_out
