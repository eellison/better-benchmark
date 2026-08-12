"""cuTile port of sum_sum_85ee747617a6: masked-LM cross-entropy backward.

Similar to sum_0aced6470e1e but with an extra pad output and column-sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


VOCAB = 30522


@ct.kernel
def _ce_backward_kernel(
    view_ptr,       # bf16 [rows, V_padded]
    arg4_ptr,       # f32  [rows, 1]
    arg5_ptr,       # f32  [rows, 1]
    arg6_ptr,       # bf16 [rows, V_padded]
    where_val_ptr,  # f32  [rows, 1]
    label_ptr,      # i64  [rows, 1]
    ne_ptr,         # b8   [rows, 1]
    sum1_ptr,       # f32  [rows, 1]
    out_ptr,        # bf16 [rows, V_padded]
    V: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    row = ct.bid(0)
    v_block = ct.bid(1)

    view = ct.load(view_ptr, index=(row, v_block), shape=(BLOCK_M, BLOCK_N))
    arg6 = ct.load(arg6_ptr, index=(row, v_block), shape=(BLOCK_M, BLOCK_N))
    where_val = ct.load(where_val_ptr, index=(row, 0), shape=(BLOCK_M, 1))
    label = ct.load(label_ptr, index=(row, 0), shape=(BLOCK_M, 1))
    ne = ct.load(ne_ptr, index=(row, 0), shape=(BLOCK_M, 1))
    arg4 = ct.load(arg4_ptr, index=(row, 0), shape=(BLOCK_M, 1))
    arg5 = ct.load(arg5_ptr, index=(row, 0), shape=(BLOCK_M, 1))
    sum_1 = ct.load(sum1_ptr, index=(row, 0), shape=(BLOCK_M, 1))

    view_f = ct.astype(view, ct.float32)
    sub = view_f - arg4
    sub_1 = sub - arg5
    # Match repro: bf16 round trip
    sub1_bf = ct.astype(sub_1, ct.bfloat16)
    sub1_f = ct.astype(sub1_bf, ct.float32)
    ex = ct.exp(sub1_f)

    cols = ct.arange(BLOCK_N, dtype=ct.int32) + v_block * BLOCK_N
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    label_i32 = ct.astype(label, ct.int32)
    match = cols_2d == label_i32
    v_valid = cols_2d < V
    match_valid = match & v_valid
    neg_one = ct.full((BLOCK_M, BLOCK_N), -1.0, dtype=ct.float32)
    zero_f = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    where_1 = ct.where(match_valid, neg_one, zero_f)

    where_2_val = ct.where(ne, where_val, ct.zeros((BLOCK_M, 1), dtype=ct.float32))
    mul = where_1 * where_2_val
    mul_bf = ct.astype(mul, ct.bfloat16)
    mul_f = ct.astype(mul_bf, ct.float32)

    sub_2 = mul_f - ex * sum_1
    add = arg6 + ct.astype(sub_2, ct.bfloat16)
    ct.store(out_ptr, index=(row, v_block), tile=add)


def _run(inputs, BLOCK_N, rows, batch_shape):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     *_shape_params) = inputs
    device = arg0_1.device

    div = arg0_1 / arg1_1
    view_labels = arg2_1.view(-1)
    ne = (view_labels != -100).view(rows, 1)
    zero_i = torch.zeros((), dtype=torch.int64, device=device)
    where_labels = torch.where(ne, view_labels.view(rows, 1), zero_i)
    where_val_full = torch.where(ne, div, torch.tensor(0.0, device=device, dtype=torch.float32))
    where_val = where_val_full.view(rows, 1)
    # sum_1 = sum(mul_bf16, [1], keepdim). mul is where_1 (-1 for label col) * where_val.
    # Since bf16 rounding of a single -where_val for the label column is exact
    # (fits in bf16 with same rounding), sum_1 = -where_val_bf16 when valid.
    where_val_bf = (-where_val).to(torch.bfloat16).to(torch.float32)
    sum_1 = torch.where(ne, where_val_bf, torch.zeros_like(where_val))

    # arg3 has stride 30528 in dim 1; make contiguous first.
    view_flat = arg3_1.contiguous().view(rows, VOCAB)
    arg6_flat = arg6_1.contiguous().view(rows, VOCAB)

    V_padded = ((VOCAB + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    padded_view = torch.zeros((rows, V_padded), device=device, dtype=torch.bfloat16)
    padded_view[:, :VOCAB].copy_(view_flat)
    padded_arg6 = torch.zeros((rows, V_padded), device=device, dtype=torch.bfloat16)
    padded_arg6[:, :VOCAB].copy_(arg6_flat)

    padded_out = torch.empty((rows, V_padded), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, ct.cdiv(V_padded, BLOCK_N), 1),
        _ce_backward_kernel,
        (padded_view, arg4_1, arg5_1, padded_arg6, where_val, where_labels, ne, sum_1,
         padded_out, VOCAB, BLOCK_N, 1),
    )

    view_4 = padded_out[:, :VOCAB].contiguous().view(rows, VOCAB)
    constant_pad = torch.nn.functional.pad(view_4, [0, 6])  # (rows, 30528)
    permute = view_4.permute(1, 0)
    sum_2 = view_4.to(torch.float32).sum(dim=0, keepdim=True)  # (1, 30522)
    view_5 = sum_2.view(-1)  # (30522,)
    conv_bf = view_5.to(torch.bfloat16)
    conv_f = conv_bf.to(torch.float32)
    return constant_pad, permute, conv_f


@oracle_impl(hardware="B200", point="3875b335", BLOCK_N=4096)
def oracle_forward_3875b335(inputs, *, BLOCK_N: int):
    return _run(inputs, BLOCK_N, rows=32768, batch_shape=(256, 128))


@oracle_impl(hardware="B200", point="7eef06a5", BLOCK_N=4096)
def oracle_forward(inputs, *, BLOCK_N: int):
    return _run(inputs, BLOCK_N, rows=16384, batch_shape=(32, 512))
