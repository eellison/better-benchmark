"""cuTile port of sum_0aced6470e1e: OPT cross-entropy backward.

For each row: cross-entropy backward with vocab=50272. Kernel computes
grad = -one_hot_target*p + softmax*p_where_valid + arg6.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


VOCAB = 50272


@ct.kernel
def _ce_backward_kernel(
    view_ptr,       # bf16 [rows, V_padded]
    arg4_ptr,       # f32  [rows, 1]  (amax)
    arg5_ptr,       # f32  [rows, 1]  (log sum)
    arg6_ptr,       # bf16 [rows, V_padded]  (accumulator)
    where_val_ptr,  # f32  [rows, 1]
    label_ptr,      # i64  [rows, 1]
    ne_ptr,         # b8   [rows, 1]
    sum1_ptr,       # f32  [rows, 1]  precomputed
    out_ptr,        # bf16 [rows, V_padded]
    V: ct.Constant[int],
    V_BASE: ct.Constant[int],
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
    ex = ct.exp(sub_1)

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

    sub_2 = mul - ex * sum_1

    add = arg6 + ct.astype(sub_2, ct.bfloat16)
    ct.store(out_ptr, index=(row, v_block), tile=add)


@oracle_impl(hardware="B200", point="1d8e61f8", BLOCK_N=4096)
def oracle_forward(inputs, *, BLOCK_N: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     *_shape_params) = inputs
    device = arg0_1.device

    # Compute label slicing and helpers on torch side.
    div = arg0_1 / arg1_1  # scalar
    slice_1 = arg2_1[:, 1:].contiguous()  # (4, 2048)
    view = slice_1.view(-1)                # (8192,)
    ne = (view != -100).view(8192, 1)
    full_zero = torch.zeros((), dtype=torch.int64, device=device)
    where_labels = torch.where(ne, view.view(8192, 1), full_zero)
    where_val_full = torch.where(ne, div, torch.tensor(0.0, device=device, dtype=torch.float32))
    where_val = where_val_full.view(8192, 1).expand(8192, 1).contiguous()

    # sum_1 = sum over vocab of where_1 * where_val = -where_val * ne (for valid labels)
    sum_1 = torch.where(ne, -where_val, torch.zeros_like(where_val))  # (8192, 1)

    rows = 8192
    view_flat = arg3_1.view(rows, VOCAB)
    arg6_flat = arg6_1.view(rows, VOCAB)

    # Pad V to BLOCK_N boundary
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
         padded_out, VOCAB, 0, BLOCK_N, 1),
    )

    view_4 = padded_out[:, :VOCAB].contiguous().view(4, 2048, VOCAB)
    view_4_flat = view_4.view(rows, VOCAB)
    permute = view_4_flat.permute(1, 0)
    return view_4_flat, permute
