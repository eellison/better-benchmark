"""cuTile port of pointwise_3124e7b9efb0: MobileBERT embedding + affine.

For each row r in [0, ROWS=BATCH*SEQ): compute affine = (activation +
word_table[token_id[r%SEQ]] + token_type_table[0]) * scale + bias.
Outputs: full[BATCH,SEQ] i64 zeros, word_emb[1,SEQ,HIDDEN] (only for r<SEQ),
token_emb[BATCH,SEQ,HIDDEN] (broadcast token_type_table[0]), affine[BATCH,SEQ,HIDDEN],
bf16_view [ROWS,HIDDEN].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 256
SEQ = 128
HIDDEN = 512
ROWS = BATCH * SEQ


@ct.kernel
def _mobilebert_kernel(
    token_ids_arr,      # i64 [SEQ]
    activations_arr,    # bf16 [ROWS, HIDDEN]
    word_table_arr,     # f32 [VOCAB, HIDDEN]
    token_type_arr,     # f32 [HIDDEN]  (row 0 of table)
    scale_arr,          # f32 [HIDDEN]
    bias_arr,           # f32 [HIDDEN]
    affine_out_arr,     # f32 [ROWS, HIDDEN]
    bf16_view_arr,      # bf16 [ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    SEQ_: ct.Constant[int],
):
    row = ct.bid(0)
    seq_idx = row - (row // SEQ_) * SEQ_

    activation = ct.load(activations_arr, index=(row, 0), shape=(1, HIDDEN_))
    activation_f = ct.astype(activation, ct.float32)

    # word_table lookup: word_table[token_ids[seq_idx], :]
    token_id_tile = ct.load(token_ids_arr, index=(seq_idx,), shape=(1,))
    token_id_scalar = ct.astype(token_id_tile, ct.int32)
    # Load row of word_table
    word = ct.load(word_table_arr, index=(0, 0), shape=(1, 1))  # placeholder
    # Actually we can't use dynamic scalar as an index directly for load. Use ct.gather instead.
    # word_table is [512, 512]. Read row using load with a dynamic index.
    # We'll use ct.load with the token_id_scalar as the outer index.
    # But ct.bid gives static, load index must be static too... Actually load
    # accepts runtime index values.
    # Try direct load:
    idx = ct.reshape(token_id_scalar, ())  # this may not work
    word = ct.load(word_table_arr, index=(idx, 0), shape=(1, HIDDEN_))
    word_f = ct.astype(word, ct.float32)

    token_type = ct.load(token_type_arr, index=(0,), shape=(HIDDEN_,))
    scale = ct.load(scale_arr, index=(0,), shape=(HIDDEN_,))
    bias = ct.load(bias_arr, index=(0,), shape=(HIDDEN_,))
    token_type_f = ct.astype(token_type, ct.float32)
    scale_f = ct.astype(scale, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    token_type_2d = ct.reshape(token_type_f, (1, HIDDEN_))
    scale_2d = ct.reshape(scale_f, (1, HIDDEN_))
    bias_2d = ct.reshape(bias_f, (1, HIDDEN_))

    add0 = activation_f + word_f
    add1 = add0 + token_type_2d
    scaled = add1 * scale_2d
    affine = scaled + bias_2d

    ct.store(affine_out_arr, index=(row, 0), tile=affine)
    ct.store(bf16_view_arr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="c404fad8", BLOCK_M=1, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    token_ids, activations, word_table, token_type_table, scale, bias, _full_shape, _view_shape, _flat_shape = inputs
    device = token_ids.device

    full = torch.zeros((BATCH, SEQ), device=device, dtype=torch.int64)
    word_embedding = torch.empty_strided(
        (1, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=word_table.device,
        dtype=torch.float32,
    )
    token_embedding = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=token_type_table.device,
        dtype=torch.float32,
    )
    affine = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=activations.device,
        dtype=torch.float32,
    )
    bf16_view = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=activations.device,
        dtype=activations.dtype,
    )

    # Pre-compute the auxiliary outputs.
    # word_embedding[0, seq, :] = word_table[token_ids[0, seq], :]
    # token_ids has shape [1, 512], we slice to [1, 128] then index
    token_ids_sliced = token_ids[0, :SEQ]
    word_embedding.view(SEQ, HIDDEN).copy_(word_table[token_ids_sliced])
    # token_embedding[:, :, :] = token_type_table[0, :] (broadcast)
    token_embedding.view(-1, HIDDEN).copy_(
        token_type_table[0:1].expand(BATCH * SEQ, HIDDEN)
    )

    stream = torch.cuda.current_stream()
    affine_2d = affine.view(ROWS, HIDDEN)
    token_type_row0 = token_type_table[0].contiguous()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _mobilebert_kernel,
        (token_ids_sliced, activations, word_table, token_type_row0,
         scale, bias, affine_2d, bf16_view, HIDDEN, SEQ),
    )
    return full, word_embedding, token_embedding, affine, bf16_view
