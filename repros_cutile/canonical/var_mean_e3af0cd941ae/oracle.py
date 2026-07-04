"""cuTile port of var_mean_e3af0cd941ae: Bart/PLBart embedding + position + LayerNorm.

For each row (batch, seq): gather token embedding (shape=[HIDDEN] from vocab),
add position embedding at index (seq + 2), round to bf16, then LayerNorm with
correction=0 (population variance), affine, output bf16.

HIDDEN can be 1024 (power of 2) or 768 (non-power-of-2). For 768 we pad to
1024 with zero and mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
POSITION_OFFSET = 2


@ct.kernel
def _embedding_layernorm_kernel(
    token_table_ptr,     # bf16 [VOCAB, BLOCK_H] (padded if HIDDEN != BLOCK_H)
    token_ids_ptr,       # i64 [rows]
    position_table_ptr,  # bf16 [POSITIONS, BLOCK_H] (padded if HIDDEN != BLOCK_H)
    weight_ptr,          # bf16 [BLOCK_H] (padded)
    bias_ptr,            # bf16 [BLOCK_H] (padded)
    out_ptr,             # bf16 [rows, BLOCK_H] (padded)
    SEQ_LEN: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    POSITION_OFFSET_: ct.Constant[int],
    EPS_: ct.Constant[float],
):
    row = ct.bid(0)
    # Load token id (scalar per row)
    token = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    # Gather one row from token table
    row_idx = ct.reshape(token, (1, 1))
    col_idx = ct.reshape(ct.arange(BLOCK_H, dtype=ct.int64), (1, BLOCK_H))
    tok_emb = ct.gather(token_table_ptr, (row_idx, col_idx))  # (1, BLOCK_H) bf16

    # Compute position from row: seq = row % SEQ_LEN, pos_idx = seq + POSITION_OFFSET.
    seq_val = row - (row // SEQ_LEN) * SEQ_LEN
    pos_idx = seq_val + POSITION_OFFSET_
    pos_emb = ct.load(position_table_ptr, index=(pos_idx, 0), shape=(1, BLOCK_H))

    tok_f = ct.astype(tok_emb, ct.float32)
    pos_f = ct.astype(pos_emb, ct.float32)
    summed_f = tok_f + pos_f
    # bf16 rounding boundary (matches Triton's !USE_INDUCTOR_NUMERICS branch)
    summed_bf = ct.astype(summed_f, ct.bfloat16)
    summed = ct.astype(summed_bf, ct.float32)

    # Mask invalid columns (BLOCK_H > HIDDEN case)
    col_idx_i32 = ct.arange(BLOCK_H, dtype=ct.int32)
    valid = ct.reshape(col_idx_i32 < HIDDEN, (1, BLOCK_H))
    zero_2d = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    summed_masked = ct.where(valid, summed, zero_2d)

    mean = ct.sum(summed_masked) * (1.0 / HIDDEN)
    centered = summed - mean
    centered_masked = ct.where(valid, centered, zero_2d)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS_)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_f = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))

    out = normalized * weight_f + bias_f
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


def _pad_last_dim(t: torch.Tensor, hidden: int, block_h: int) -> torch.Tensor:
    if hidden == block_h:
        return t
    if t.dim() == 1:
        padded = torch.zeros((block_h,), device=t.device, dtype=t.dtype)
        padded[:hidden] = t
        return padded
    # 2D: pad last dim
    other = int(t.shape[0])
    padded = torch.zeros((other, block_h), device=t.device, dtype=t.dtype)
    padded[:, :hidden] = t
    return padded


@oracle_impl(hardware="B200", point="dacf9422", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="90ede91d", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    padded_token_table = _pad_last_dim(token_table, hidden, BLOCK_H)
    padded_position_table = _pad_last_dim(position_table, hidden, BLOCK_H)
    padded_weight = _pad_last_dim(weight, hidden, BLOCK_H)
    padded_bias = _pad_last_dim(bias, hidden, BLOCK_H)

    padded_out = torch.empty((rows, BLOCK_H), device=token_table.device, dtype=torch.bfloat16)

    ids_flat = token_ids.view(rows).to(torch.int64)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _embedding_layernorm_kernel,
        (padded_token_table, ids_flat, padded_position_table,
         padded_weight, padded_bias, padded_out,
         seq_len, hidden, BLOCK_H, POSITION_OFFSET, EPS),
    )

    out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    out.view(rows, hidden).copy_(padded_out[:, :hidden])

    def _as_shape(shape):
        return tuple(int(d) for d in shape)

    return (
        out,
        out.view(_as_shape(shape0)),
        out.view(_as_shape(shape1)),
        out.view(_as_shape(shape2)),
    )
