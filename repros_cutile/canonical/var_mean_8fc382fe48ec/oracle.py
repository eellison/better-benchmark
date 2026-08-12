"""cuTile port of var_mean_8fc382fe48ec: M2M100 embed + LayerNorm.

Per row:
  token = embed_table[token_id, :]   (bf16)
  position_id = int64(token_pos * pos_mask) + 1
  position = position_table[position_id, :]  (bf16)
  add = bf16(bf16(token * 32.0).f32 + position.f32)   -- residual add w/ bf16 rounding
  out = layernorm(add) then affine with weight/bias -> bf16
Returns: (add_out, view0, view1, view2) where views alias out flat [8192, 1024].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
EMBED_SCALE = 32.0
HIDDEN = 1024


@ct.kernel
def _m2m100_kernel(
    token_table,     # bf16 [VOCAB, HIDDEN]
    token_ids,       # i64  [ROWS]
    positions,       # i64  [ROWS]
    position_mask,   # i32  [ROWS]
    position_table,  # bf16 [POS, HIDDEN]
    weight,          # bf16 [HIDDEN]
    bias,            # bf16 [HIDDEN]
    add_out,         # bf16 [ROWS, HIDDEN]
    final_out,       # bf16 [ROWS, HIDDEN]
    HIDDEN_c: ct.Constant[int],
):
    row = ct.bid(0)
    # Load per-row scalars
    token_id = ct.load(token_ids, index=(row,), shape=(1,))
    pos = ct.load(positions, index=(row,), shape=(1,))
    pmask = ct.load(position_mask, index=(row,), shape=(1,))
    # position_id = int64(int32(pos) * pmask) + 1
    pos_i32 = ct.astype(pos, ct.int32)
    prod_i32 = pos_i32 * pmask
    pos_id = ct.astype(prod_i32, ct.int64) + ct.full((1,), 1, dtype=ct.int64)

    # Gather rows via ct.gather over a 1D index tile
    col_idx = ct.arange(HIDDEN_c, dtype=ct.int64)
    col_idx_2d = ct.reshape(col_idx, (1, HIDDEN_c))
    token_id_2d = ct.reshape(token_id, (1, 1))
    pos_id_2d = ct.reshape(pos_id, (1, 1))
    token = ct.gather(token_table, (token_id_2d, col_idx_2d))
    position = ct.gather(position_table, (pos_id_2d, col_idx_2d))

    token_f = ct.astype(token, ct.float32)
    position_f = ct.astype(position, ct.float32)
    scaled = token_f * EMBED_SCALE
    scaled_bf = ct.astype(scaled, ct.bfloat16)
    add_f = ct.astype(scaled_bf, ct.float32) + position_f
    add_bf = ct.astype(add_f, ct.bfloat16)
    ct.store(add_out, index=(row, 0), tile=add_bf)

    x = ct.astype(add_bf, ct.float32)
    mean = ct.sum(x) * (1.0 / HIDDEN_c)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_c)
    invstd = ct.rsqrt(variance + EPS)
    weight_tile = ct.load(weight, index=(0,), shape=(HIDDEN_c,))
    bias_tile = ct.load(bias, index=(0,), shape=(HIDDEN_c,))
    weight_f = ct.astype(weight_tile, ct.float32)
    bias_f = ct.astype(bias_tile, ct.float32)
    normalized = centered * invstd
    weight_2d = ct.reshape(weight_f, (1, HIDDEN_c))
    bias_2d = ct.reshape(bias_f, (1, HIDDEN_c))
    affine = normalized * weight_2d + bias_2d
    ct.store(final_out, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="929234c0")
def oracle_forward(inputs):
    (
        token_table,
        token_ids,
        positions,
        position_mask,
        position_table,
        weight,
        bias,
        add_shape,
        view_shape0,
        view_shape1,
        view_shape2,
    ) = inputs

    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len
    base_shape = _as_shape(add_shape)
    base_stride = (seq_len * hidden, hidden, 1)

    add_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    # Flatten to [rows, hidden] for kernel
    add_out_2d = add_out.view(rows, hidden)
    final_out_2d = final_out.view(rows, hidden)
    token_ids_1d = token_ids.view(rows)
    positions_1d = positions.view(rows)
    position_mask_1d = position_mask.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _m2m100_kernel,
        (token_table, token_ids_1d, positions_1d, position_mask_1d,
         position_table, weight, bias, add_out_2d, final_out_2d, hidden),
    )

    return (
        add_out,
        final_out.view(_as_shape(view_shape0)),
        final_out.view(_as_shape(view_shape1)),
        final_out.view(_as_shape(view_shape2)),
    )
