"""cuTile port of var_mean_cf0dfeba0347: GPT-Neo embedding + LayerNorm.

For each row r in [0, batch*seq):
  seq_index = r % seq
  token_id = token_ids[r]
  token = token_table[token_id]
  position = position_table[seq_index]
  x = token + position
  LayerNorm (f32) over hidden with eps=1e-5.
  Output: token embedding (f32 [B, S, H]), position_ids (i64 [1, S]),
          mask (bool [B, S] all False), position embedding (f32 [1, S, H]),
          add (f32 [B, S, H]), mean (f32 [B, S, 1]), invstd (f32 [B, S, 1]),
          bf16 [4096, 2048] view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _gptneo_embedding_layernorm_kernel(
    token_table_ptr,      # f32 (V, H)
    token_ids_ptr,        # i64 (rows,)
    position_table_ptr,   # f32 (S, H)
    weight_ptr,           # f32 (H,)
    bias_ptr,             # f32 (H,)
    token_out_ptr,        # f32 (rows, H)
    add_out_ptr,          # f32 (rows, H)
    mean_out_ptr,         # f32 (rows,)
    invstd_out_ptr,       # f32 (rows,)
    bf16_out_ptr,         # bf16 (rows, H)
    BLOCK_H: ct.Constant[int],
    SEQ_: ct.Constant[int],
):
    row = ct.bid(0)

    token_id_tile = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    token_id_scalar = ct.reshape(token_id_tile, ())
    seq_index_scalar = row % SEQ_

    token = ct.load(token_table_ptr, index=(token_id_scalar, 0), shape=(1, BLOCK_H))
    position = ct.load(position_table_ptr, index=(seq_index_scalar, 0), shape=(1, BLOCK_H))

    token_f = ct.astype(token, ct.float32)
    position_f = ct.astype(position, ct.float32)
    x = token_f + position_f

    ct.store(token_out_ptr, index=(row, 0), tile=token_f)
    ct.store(add_out_ptr, index=(row, 0), tile=x)

    mean = ct.sum(x, axis=1, keepdims=True) * (1.0 / BLOCK_H)
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * (1.0 / BLOCK_H)
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.reshape(weight, (1, BLOCK_H))
    bias_f = ct.reshape(bias, (1, BLOCK_H))
    y = centered * invstd * weight_f + bias_f
    ct.store(bf16_out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))

    ct.store(mean_out_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_out_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))


@ct.kernel
def _position_row_kernel(
    position_table_ptr,   # f32 (S_max, H)
    position_out_ptr,     # f32 (S, H)
    BLOCK_H: ct.Constant[int],
):
    s = ct.bid(0)
    pos = ct.load(position_table_ptr, index=(s, 0), shape=(1, BLOCK_H))
    ct.store(position_out_ptr, index=(s, 0), tile=pos)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="2b53f84f", BLOCK_H=2048)
def oracle_forward(inputs, *, BLOCK_H: int):
    token_table, token_ids, position_table, weight, bias, _expand_shape, view_shape = inputs
    batch = int(token_ids.shape[0])
    seq = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq

    token_out = torch.empty_strided((batch, seq, hidden), (seq * hidden, hidden, 1),
                                    device=token_table.device, dtype=torch.float32)
    add_out = torch.empty_strided((batch, seq, hidden), (seq * hidden, hidden, 1),
                                  device=token_table.device, dtype=torch.float32)
    mean = torch.empty_strided((batch, seq, 1), (seq, 1, 1),
                               device=token_table.device, dtype=torch.float32)
    invstd = torch.empty_strided((batch, seq, 1), (seq, 1, 1),
                                 device=token_table.device, dtype=torch.float32)
    bf16_base = torch.empty_strided((batch, seq, hidden), (seq * hidden, hidden, 1),
                                    device=token_table.device, dtype=torch.bfloat16)

    token_ids_1d = token_ids.view(rows)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _gptneo_embedding_layernorm_kernel,
        (
            token_table, token_ids_1d, position_table, weight, bias,
            token_out.view(rows, hidden), add_out.view(rows, hidden),
            mean.view(rows), invstd.view(rows),
            bf16_base.view(rows, hidden),
            hidden, seq,
        ),
    )

    # Position embedding side output — [1, S, H] = position_table[0:S]
    position_out = torch.empty_strided((1, seq, hidden), (seq * hidden, hidden, 1),
                                       device=token_table.device, dtype=torch.float32)
    ct.launch(
        stream,
        (seq, 1, 1),
        _position_row_kernel,
        (position_table, position_out.view(seq, hidden), hidden),
    )

    # position_ids [1, S] = [0..S-1]
    position_ids = torch.arange(seq, dtype=torch.int64, device=token_table.device).view(1, seq)
    # mask [B, S] all False
    mask_out = torch.zeros((batch, seq), device=token_table.device, dtype=torch.bool)

    return (
        token_out,
        position_ids,
        mask_out,
        position_out,
        add_out,
        mean,
        invstd,
        bf16_base.view(_shape_tuple(view_shape)),
    )
