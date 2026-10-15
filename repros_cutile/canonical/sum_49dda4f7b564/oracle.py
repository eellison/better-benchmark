"""cuTile port of sum_49dda4f7b564: XGLM cross-entropy backward.

Row kernel: for each of 4096 tokens, sum over 256008 vocab positions.
Uses cuTile's default RTNE for f32 arithmetic (no inline PTX needed).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_ROWS = 4096
VOCAB = 256008


@ct.kernel
def _xent_backward_kernel(
    logits_ptr,      # bf16 [rows, VOCAB]
    max_ptr,         # f32 [rows]
    logsumexp_ptr,   # f32 [rows]
    label_ptr,       # i64 [rows]
    scale_ptr,       # f32 [rows]
    ne_ptr,          # bool [rows]
    grad_out_ptr,    # bf16 [rows, VOCAB]
    out_ptr,         # bf16 [rows, VOCAB]
    VOCAB_: ct.Constant[int],
    BLOCK_V: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)

    # Load a tile of BLOCK_V vocab positions from row `row`.
    logits = ct.load(
        logits_ptr, index=(row, col_block), shape=(1, BLOCK_V),
        padding_mode=ct.PaddingMode.ZERO,
    )
    grad_out = ct.load(
        grad_out_ptr, index=(row, col_block), shape=(1, BLOCK_V),
        padding_mode=ct.PaddingMode.ZERO,
    )
    logits_f = ct.astype(logits, ct.float32)

    row_max = ct.load(max_ptr, index=(row,), shape=(1,))
    row_lse = ct.load(logsumexp_ptr, index=(row,), shape=(1,))
    label_val = ct.load(label_ptr, index=(row,), shape=(1,))
    scale_val = ct.load(scale_ptr, index=(row,), shape=(1,))
    ne_val = ct.load(ne_ptr, index=(row,), shape=(1,))

    # softmax = exp(logits - row_max - lse)
    row_max_2d = ct.reshape(row_max, (1, 1))
    row_lse_2d = ct.reshape(row_lse, (1, 1))
    label_2d = ct.reshape(label_val, (1, 1))
    scale_2d = ct.reshape(scale_val, (1, 1))
    ne_2d = ct.reshape(ne_val, (1, 1))
    softmax = ct.exp(logits_f - row_max_2d - row_lse_2d)

    # Column indices
    col_offset = col_block * BLOCK_V
    col_idx = ct.arange(BLOCK_V, dtype=ct.int64) + col_offset
    col_idx_2d = ct.reshape(col_idx, (1, BLOCK_V))
    is_label_col = col_idx_2d == label_2d
    # where_1_2d: -1 if label, else 0
    where_1 = ct.where(is_label_col, -1.0, 0.0)
    # where_2 (per row): scale if ne, else 0
    where_2 = ct.where(ne_2d, scale_2d, 0.0)
    mul = where_1 * where_2  # shape (1, BLOCK_V)
    # sum_1 = sum(mul, axis=1) but only for this vocab tile — we need the
    # full-row sum_1. Since where_1 is -1 only at the label column and 0
    # elsewhere, and where_2 is scalar per-row, sum_1 = -where_2 if the label
    # falls in this tile, else 0.
    # We'll rely on the outer torch stage to compute sum_1 once per row.
    # Here we just apply grad computation given a precomputed sum_1.
    # But we haven't loaded sum_1 yet. Let's just do:
    #   grad = mul - softmax * sum_1
    # We need sum_1 as per-row scalar; load it too.
    # Move that above:
    pass_val = mul
    grad_partial = pass_val  # placeholder; we'll finalize below


@ct.kernel
def _xent_backward_final_kernel(
    logits_ptr,      # bf16 [rows, VOCAB]
    max_ptr,         # f32 [rows]
    logsumexp_ptr,   # f32 [rows]
    label_ptr,       # i64 [rows]
    scale_ptr,       # f32 [rows]
    ne_ptr,          # bool [rows]
    sum1_ptr,        # f32 [rows]  (precomputed = where_2 mask contribution)
    grad_out_ptr,    # bf16 [rows, VOCAB]
    out_ptr,         # bf16 [rows, VOCAB]
    VOCAB_: ct.Constant[int],
    BLOCK_V: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)

    logits = ct.load(
        logits_ptr, index=(row, col_block), shape=(1, BLOCK_V),
        padding_mode=ct.PaddingMode.ZERO,
    )
    grad_out = ct.load(
        grad_out_ptr, index=(row, col_block), shape=(1, BLOCK_V),
        padding_mode=ct.PaddingMode.ZERO,
    )
    logits_f = ct.astype(logits, ct.float32)

    row_max = ct.load(max_ptr, index=(row,), shape=(1,))
    row_lse = ct.load(logsumexp_ptr, index=(row,), shape=(1,))
    label_val = ct.load(label_ptr, index=(row,), shape=(1,))
    scale_val = ct.load(scale_ptr, index=(row,), shape=(1,))
    ne_val = ct.load(ne_ptr, index=(row,), shape=(1,))
    sum1 = ct.load(sum1_ptr, index=(row,), shape=(1,))

    row_max_2d = ct.reshape(row_max, (1, 1))
    row_lse_2d = ct.reshape(row_lse, (1, 1))
    label_2d = ct.reshape(label_val, (1, 1))
    scale_2d = ct.reshape(scale_val, (1, 1))
    ne_2d = ct.reshape(ne_val, (1, 1))
    sum1_2d = ct.reshape(sum1, (1, 1))
    # softmax(logits) = exp(logits - max - lse) where lse = logsumexp of the row
    softmax = ct.exp(logits_f - row_max_2d - row_lse_2d)

    col_offset = col_block * BLOCK_V
    col_idx = ct.arange(BLOCK_V, dtype=ct.int64) + col_offset
    col_idx_2d = ct.reshape(col_idx, (1, BLOCK_V))
    is_label_col = col_idx_2d == label_2d
    where_1 = ct.where(is_label_col, -1.0, 0.0)
    where_2 = ct.where(ne_2d, scale_2d, 0.0)
    mul = where_1 * where_2
    grad = mul - softmax * sum1_2d
    grad_bf = ct.astype(grad, ct.bfloat16)
    add_bf = ct.astype(
        ct.astype(grad_out, ct.float32) + ct.astype(grad_bf, ct.float32),
        ct.bfloat16,
    )
    ct.store(out_ptr, index=(row, col_block), tile=add_bf)


@oracle_impl(hardware="B200", point="771c693d", BLOCK_V=1024)
def oracle_forward(inputs, *, BLOCK_V: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        _shape0, _shape1, _shape2, _shape3, shape4,
    ) = inputs
    device = arg3_1.device
    rows = N_ROWS
    vocab = VOCAB
    out_shape = tuple(int(d) for d in shape4)

    # Compute per-row precursor values on the Python side.
    div = arg0_1 / arg1_1  # f32 scalar
    slice_pos = arg2_1[:, 1:].contiguous()
    label = slice_pos.view(-1).unsqueeze(-1)  # [4096, 1]
    ne = (label != -100)  # bool [4096, 1]
    where = torch.where(ne, label, torch.tensor(0, device=device, dtype=torch.int64))
    where_2 = torch.where(ne, div.expand_as(label.to(torch.float32)), torch.tensor(0.0, device=device, dtype=torch.float32))
    # sum_1 = sum over vocab of where_1 * where_2, where where_1 = -1 iff col==label else 0
    # sum_1 = -where_2 (per-row scalar) if label is a valid vocab index
    sum_1 = -where_2  # shape [4096, 1]

    # Reshape to per-row scalars
    label_1d = where.view(-1)  # [4096]
    ne_1d = ne.view(-1)
    scale_1d = torch.where(ne_1d, div.to(torch.float32).expand(rows), torch.zeros(rows, device=device, dtype=torch.float32))
    sum_1_1d = sum_1.view(-1)

    # Row max and logsumexp from arg4, arg5:
    # In the Triton oracle: sub = view_2 - arg4 - arg5; exp(sub) is per-row softmax
    # so arg4 is row_max, arg5 is logsumexp of the shifted logits.
    row_max = arg4_1.view(-1).to(torch.float32)
    row_lse = arg5_1.view(-1).to(torch.float32)

    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    logits_2d = arg3_1.contiguous().view(rows, vocab)
    grad_out_2d = arg6_1.contiguous().view(rows, vocab)
    out_2d = out.view(rows, vocab)

    stream = torch.cuda.current_stream()
    n_col_blocks = ct.cdiv(vocab, BLOCK_V)
    ct.launch(
        stream, (rows, n_col_blocks, 1), _xent_backward_final_kernel,
        (logits_2d, row_max, row_lse, label_1d, scale_1d, ne_1d, sum_1_1d,
         grad_out_2d, out_2d, vocab, BLOCK_V),
    )
    return out, out.permute(1, 0)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))
