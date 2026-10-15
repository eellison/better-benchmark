"""cuTile port of amax_sum_sum_2df7bcca6b5d: Electra CE loss (cross-entropy).

Row-wise logsumexp over VOCAB=30522 columns, then gather label prob, then
mean over valid rows. VOCAB is non-power-of-2 so we pad to 32768 and mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


VOCAB = 30522
PADDED = 32768  # power of 2 >= VOCAB


@ct.kernel
def _logsumexp_gather_kernel(
    logits_ptr,     # bf16 [rows, PADDED] padded, invalid cols = -inf sentinel
    labels_ptr,     # i64 [rows]  (labels, unclamped)
    ne_mask_ptr,    # b8  [rows]  (labels != -100)
    per_row_ptr,    # f32 [rows]  output: nll * ne
    V: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(logits_ptr, index=(row, 0), shape=(BLOCK_M, BLOCK_N))
    x_f = ct.astype(x_bf, ct.float32)

    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    col_valid = cols < V
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_N))
    neg_inf = ct.full((BLOCK_M, BLOCK_N), -float("inf"), dtype=ct.float32)
    x_masked = ct.where(col_valid_2d, x_f, neg_inf)

    amax_val = ct.max(x_masked, axis=1, keepdims=True)
    sub = x_f - amax_val
    ex = ct.exp(sub)
    zero = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    ex_m = ct.where(col_valid_2d, ex, zero)
    sum_val = ct.sum(ex_m, axis=1, keepdims=True)
    log_sum = ct.log(sum_val)
    # sub_1 = sub - log
    # gather label position
    label = ct.load(labels_ptr, index=(row,), shape=(BLOCK_M,))
    ne_mask = ct.load(ne_mask_ptr, index=(row,), shape=(BLOCK_M,))
    zero_i = ct.zeros((BLOCK_M,), dtype=ct.int64)
    label_clamped = ct.where(ne_mask, label, zero_i)
    # We need sub_1[row, label] = sub[row, label] - log_sum[row].
    # In tile-space, we need to select an arbitrary column. Since BLOCK_M=1,
    # we compute the sub tile and compare cols == label_clamped[0].
    label_col = ct.astype(label_clamped, ct.int32)
    label_col_2d = ct.reshape(label_col, (1, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    matches = cols_2d == label_col_2d  # (1, BLOCK_N)
    # sub[row, label] extracted via sum(sub*matches, axis=1)
    sub1 = sub - log_sum
    gathered = ct.sum(ct.where(matches, sub1, zero), axis=1)  # (1,)
    neg = 0.0 - gathered
    ne_mask_f = ct.astype(ne_mask, ct.float32)
    per_row = neg * ne_mask_f
    ct.store(per_row_ptr, index=(row,), tile=per_row)


@oracle_impl(hardware="B200", point="a7052938", BLOCK_N=32768)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, *_shape_params = inputs
    device = arg0_1.device

    # constant_pad_nd -> slice -> clone -> view: reproduce
    constant_pad = torch.nn.functional.pad(arg0_1, [0, 1], value=-100)
    slice_1 = constant_pad[:, 1:]
    labels_view = slice_1.contiguous().view(-1)  # [32768]
    ne = labels_view != -100

    # slice_2: bf16[32768, 30522]
    logits_slice = arg1_1[:, :VOCAB].contiguous()
    view_1 = logits_slice.view(64, 512, VOCAB)

    rows = 32768
    # Pad logits to (rows, BLOCK_N)
    padded_logits = torch.empty((rows, BLOCK_N), device=device, dtype=torch.bfloat16)
    padded_logits[:, :VOCAB].copy_(logits_slice)
    # Fill padding with anything (we mask in kernel).
    padded_logits[:, VOCAB:].fill_(0)

    per_row = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _logsumexp_gather_kernel,
        (padded_logits, labels_view, ne, per_row, VOCAB, BLOCK_N, 1),
    )

    sum_2 = per_row.sum()
    sum_3 = ne.sum().to(torch.float32)
    div = sum_2 / sum_3
    return view_1, div
