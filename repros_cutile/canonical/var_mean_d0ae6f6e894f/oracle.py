"""cuTile port of var_mean_d0ae6f6e894f: LayoutLM embedding+LayerNorm.

Row kernel (one row per program, HIDDEN=768 padded to BLOCK_H=1024):
  Gather word_embedding and position_embedding for this row.
  Sequentially add x_pos + y_pos + x_pos + y_pos + h_pos + w_pos + tt
  (all sourced from full=0 → index 0 of their respective tables) — with a
  bf16 rounding boundary at each add step to match the eager bf16 add
  operator chain.
  Then fp32 var_mean over the first 768 cols, rsqrt(var + 1e-12), affine
  (weight*norm + bias), cast to bf16, store first 768 cols.

We pad tables to BLOCK_H=1024 with zeros so cuTile power-of-2 shape rules
apply, then slice back after the kernel.

`_USE_INDUCTOR_NUMERICS` flips ON when the oracle runs inside a CUDA graph
capture (matches the Triton reference).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


_USE_INDUCTOR_NUMERICS = False
BLOCK_H = 1024


@ct.kernel
def _layoutlm_ln_kernel(
    word_table_ptr,       # bf16 [VOCAB, BLOCK_H] padded
    word_ids_ptr,         # i64 [ROWS]
    position_table_ptr,   # bf16 [POS_LEN, BLOCK_H] padded
    position_ids_ptr,     # i64 [SEQ_LEN]
    x_row_ptr,            # bf16 [BLOCK_H] padded
    y_row_ptr,            # bf16 [BLOCK_H] padded
    h_row_ptr,            # bf16 [BLOCK_H] padded
    w_row_ptr,            # bf16 [BLOCK_H] padded
    tt_row_ptr,           # bf16 [BLOCK_H] padded
    weight_ptr,           # bf16 [BLOCK_H] padded
    bias_ptr,             # bf16 [BLOCK_H] padded
    out_ptr,              # bf16 [ROWS, BLOCK_H] padded
    HIDDEN: ct.Constant[int],
    SEQ_LEN: ct.Constant[int],
    EPS: ct.Constant[float],
    USE_INDUCTOR: ct.Constant[bool],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    seq = row - (row // SEQ_LEN) * SEQ_LEN

    word_id_1 = ct.load(word_ids_ptr, index=(row,), shape=(1,))
    position_id_1 = ct.load(position_ids_ptr, index=(seq,), shape=(1,))
    word_id = ct.reshape(word_id_1, ())
    position_id = ct.reshape(position_id_1, ())

    word_bf = ct.load(word_table_ptr, index=(word_id, 0), shape=(1, BLOCK_H_))
    position_bf = ct.load(position_table_ptr, index=(position_id, 0), shape=(1, BLOCK_H_))
    x_row = ct.reshape(ct.load(x_row_ptr, index=(0,), shape=(BLOCK_H_,)), (1, BLOCK_H_))
    y_row = ct.reshape(ct.load(y_row_ptr, index=(0,), shape=(BLOCK_H_,)), (1, BLOCK_H_))
    h_row = ct.reshape(ct.load(h_row_ptr, index=(0,), shape=(BLOCK_H_,)), (1, BLOCK_H_))
    w_row = ct.reshape(ct.load(w_row_ptr, index=(0,), shape=(BLOCK_H_,)), (1, BLOCK_H_))
    tt_row = ct.reshape(ct.load(tt_row_ptr, index=(0,), shape=(BLOCK_H_,)), (1, BLOCK_H_))

    if USE_INDUCTOR:
        x = ct.astype(word_bf, ct.float32) + ct.astype(position_bf, ct.float32)
        x = x + ct.astype(x_row, ct.float32)
        x = x + ct.astype(y_row, ct.float32)
        x = x + ct.astype(x_row, ct.float32)
        x = x + ct.astype(y_row, ct.float32)
        x = x + ct.astype(h_row, ct.float32)
        x = x + ct.astype(w_row, ct.float32)
        x = x + ct.astype(tt_row, ct.float32)
    else:
        # Chain of bf16-boundary adds: each add rounds through bf16.
        def _add_bf(acc_bf, other_bf):
            return ct.astype(
                ct.astype(acc_bf, ct.float32) + ct.astype(other_bf, ct.float32),
                ct.bfloat16,
            )
        acc = _add_bf(word_bf, position_bf)
        acc = _add_bf(acc, x_row)
        acc = _add_bf(acc, y_row)
        acc = _add_bf(acc, x_row)
        acc = _add_bf(acc, y_row)
        acc = _add_bf(acc, h_row)
        acc = _add_bf(acc, w_row)
        acc = _add_bf(acc, tt_row)
        x = ct.astype(acc, ct.float32)

    # Mask valid cols for reductions.
    cols = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_valid_1d = cols < HIDDEN
    col_valid = ct.reshape(col_valid_1d, (1, BLOCK_H_))
    x_masked = ct.where(col_valid, x, 0.0)
    mean = ct.sum(x_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_valid, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_,))
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H_))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H_))
    affine = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="771dc86a")
def oracle_forward(inputs):
    global _USE_INDUCTOR_NUMERICS
    (
        word_table,
        word_ids,
        position_table,
        position_ids,
        x_position_table,
        y_position_table,
        h_position_table,
        w_position_table,
        token_type_table,
        weight,
        bias,
        _shape0,
        _shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    batch = int(word_ids.shape[0])
    seq_len = int(word_ids.shape[1])
    hidden = int(word_table.shape[1])
    rows = batch * seq_len

    use_inductor = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor = True

    device = word_table.device

    def _pad_2d(t, block):
        pad_shape = (t.shape[0], block)
        out = torch.zeros(pad_shape, device=device, dtype=t.dtype)
        out[:, :hidden] = t
        return out

    def _pad_1d(t, block):
        out = torch.zeros((block,), device=device, dtype=t.dtype)
        out[:hidden] = t
        return out

    word_padded = _pad_2d(word_table, BLOCK_H)
    position_padded = _pad_2d(position_table, BLOCK_H)
    weight_padded = _pad_1d(weight, BLOCK_H)
    bias_padded = _pad_1d(bias, BLOCK_H)

    x_row_padded = _pad_1d(x_position_table[0].contiguous(), BLOCK_H)
    y_row_padded = _pad_1d(y_position_table[0].contiguous(), BLOCK_H)
    h_row_padded = _pad_1d(h_position_table[0].contiguous(), BLOCK_H)
    w_row_padded = _pad_1d(w_position_table[0].contiguous(), BLOCK_H)
    tt_row_padded = _pad_1d(token_type_table[0].contiguous(), BLOCK_H)

    out_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layoutlm_ln_kernel,
        (
            word_padded,
            word_ids.reshape(-1).contiguous(),
            position_padded,
            position_ids.reshape(-1).contiguous(),
            x_row_padded,
            y_row_padded,
            h_row_padded,
            w_row_padded,
            tt_row_padded,
            weight_padded,
            bias_padded,
            out_padded,
            hidden,
            seq_len,
            1.0e-12,
            use_inductor,
            BLOCK_H,
        ),
    )

    out = out_padded[:, :hidden].contiguous().view(batch, seq_len, hidden)

    return (
        out,
        out.view(_shape_tuple(shape2)),
        out.view(_shape_tuple(shape3)),
        out.view(_shape_tuple(shape4)),
    )
