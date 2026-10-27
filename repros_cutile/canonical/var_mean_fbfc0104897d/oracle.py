"""cuTile port of var_mean_fbfc0104897d (NEW_PATTERN): Longformer inference
embedding + LayerNorm — per row: gather word/position/global embeddings, bf16
add chain, HIDDEN=768 mean/var, affine bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _longformer_embedding_layernorm_kernel(
    word_table_ptr,     # bf16 [V, HIDDEN]
    word_ids_ptr,       # i64 [ROWS]
    pos_seed_ptr,       # i64 [ROWS]
    pos_mask_ptr,       # i32 [ROWS]
    position_table_ptr, # bf16 [4098, HIDDEN]
    global_table_ptr,   # bf16 [HIDDEN]
    weight_ptr,         # bf16 [HIDDEN]
    bias_ptr,           # bf16 [HIDDEN]
    out_ptr,            # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_H, dtype=ct.int64)
    col_mask = cols < HIDDEN
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))
    cols_2d = ct.reshape(cols, (1, BLOCK_H))

    word_id = ct.load(word_ids_ptr, index=(row,), shape=(1,))
    pos_seed = ct.astype(
        ct.load(pos_seed_ptr, index=(row,), shape=(1,)), ct.int32
    )
    pos_mask = ct.load(pos_mask_ptr, index=(row,), shape=(1,))
    position_id = ct.astype(pos_seed * pos_mask, ct.int64) + 1
    word_id_2d = ct.reshape(word_id, (1, 1))
    pos_id_2d = ct.reshape(position_id, (1, 1))

    word = ct.astype(
        ct.gather(word_table_ptr, (word_id_2d, cols_2d),
                  mask=col_mask_2d, padding_value=ct.bfloat16(0)),
        ct.float32,
    )
    position = ct.astype(
        ct.gather(position_table_ptr, (pos_id_2d, cols_2d),
                  mask=col_mask_2d, padding_value=ct.bfloat16(0)),
        ct.float32,
    )
    global_token = ct.astype(
        ct.load(global_table_ptr, index=(0,), shape=(BLOCK_H,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    global_2d = ct.reshape(global_token, (1, BLOCK_H))

    add1_bf16 = ct.astype(word + position, ct.bfloat16)
    add1_f = ct.astype(add1_bf16, ct.float32)
    x_bf16 = ct.astype(add1_f + global_2d, ct.bfloat16)
    x = ct.astype(x_bf16, ct.float32)

    x_masked = ct.where(col_mask_2d, x, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine_bf16 = ct.astype(normalized * weight_2d + bias_2d, ct.bfloat16)

    # Scatter to correct HIDDEN=768 positions (not power-of-2)
    col_idx = ct.arange(BLOCK_H, dtype=ct.int64)
    row_idx = ct.full(shape=(BLOCK_H,), fill_value=row, dtype=ct.int64)
    ct.scatter(
        out_ptr, (row_idx, col_idx),
        ct.reshape(affine_bf16, (BLOCK_H,)),
        mask=col_mask,
    )


@oracle_impl(hardware="B200", point="e3ff68de", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        word_table,
        word_ids,
        pos_seed,
        pos_mask,
        position_table,
        global_table,
        weight,
        bias,
        _shape_param_0,
    ) = inputs
    batch = int(word_ids.shape[0])
    seq_len = int(word_ids.shape[1])
    hidden = int(weight.shape[0])
    rows = batch * seq_len
    out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )
    out_2d = out.view(rows, hidden)
    word_ids_flat = word_ids.view(rows)
    pos_seed_flat = pos_seed.view(rows)
    pos_mask_flat = pos_mask.view(rows)
    global_flat = global_table.view(hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _longformer_embedding_layernorm_kernel,
        (word_table, word_ids_flat, pos_seed_flat, pos_mask_flat,
         position_table, global_flat, weight, bias, out_2d, hidden, BLOCK_H),
    )
    return out
