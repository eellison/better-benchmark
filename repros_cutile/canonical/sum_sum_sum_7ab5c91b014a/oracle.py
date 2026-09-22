"""cuTile port of sum_sum_sum_7ab5c91b014a: BERT LN backward.

Fair port: 2 kernels matching Triton structure.
Kernel 1 (`_row_partials_kernel`): row-tiled LN backward. Writes returned
add_3 (f32) and view_3 (bf16 = add_3.bf16 * (arg5.bf16 * 1.1111...).bf16).
Accumulates partial column sums for x (view_1), x*normed (view_2), and
bf16(dropout) (convert_element_type_4).
Kernel 2 (`_finalize_partials`): finalizes column sums with a bf16-round for
the dropout sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6
ROW_BACKWARD_SCALE = 0.002607561929595828
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _row_partials_kernel(
    grad_ptr,      # bf16 (ROWS, HIDDEN)
    weight_ptr,    # f32 (HIDDEN,)
    residual_ptr,  # f32 (ROWS, HIDDEN)
    denom_base_ptr,# f32 (ROWS,)
    prev_grad_ptr, # f32 (ROWS, HIDDEN)
    keep_ptr,      # bool (ROWS, HIDDEN)
    add_3_ptr,     # f32 (ROWS, HIDDEN)
    view_3_ptr,    # bf16 (ROWS, HIDDEN)
    partials_ptr,  # f32 (ROWS_, 3, BLOCK_H)  — one per row (ROWS_PER_GROUP=1)
    HIDDEN_: ct.Constant[int],
    INV_HIDDEN_F: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H))
    zero_2d = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))

    # Naming: Triton's `x` == arg0 (grad_bf) as f32; Triton's `dy` == arg2 (f32);
    # Triton's `residual` == arg4 (prev_grad). We keep local Triton-style names.
    grad_bf = ct.load(grad_ptr, index=(row, 0), shape=(1, BLOCK_H),
                      padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(grad_bf, ct.float32)
    dy = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                 padding_mode=ct.PaddingMode.ZERO)
    prev_grad = ct.load(prev_grad_ptr, index=(row, 0), shape=(1, BLOCK_H),
                        padding_mode=ct.PaddingMode.ZERO)
    denom_base_scalar = ct.load(denom_base_ptr, index=(row,), shape=(1,))
    denom_base = ct.reshape(denom_base_scalar, (1, 1))
    denom = denom_base + EPS

    mul = weight_2d * dy
    div_ = mul / denom
    div_1 = div_ / denom
    mul_1 = (0.0 - x) * div_1
    mul_1_m = ct.where(col_mask, mul_1, zero_2d)
    sum_2 = ct.sum(mul_1_m, axis=1, keepdims=True)

    div_2 = x / denom
    mul_2 = div_2 * weight_2d
    mul_3 = div_2 * dy  # x*dy/denom -> for view_2 (mul_3 sum)
    neg_mul2 = 0.0 - mul_2
    neg_mul2_m = ct.where(col_mask, neg_mul2, zero_2d)
    sum_4 = ct.sum(neg_mul2_m, axis=1, keepdims=True)

    add_1 = prev_grad + mul_2
    mul_4 = denom_base * 2.0
    div_3 = sum_2 / mul_4
    is_zero = denom_base == 0.0
    where_val = ct.where(is_zero, ct.full((1, 1), 0.0, dtype=ct.float32), div_3)
    mul_5 = where_val * ROW_BACKWARD_SCALE
    mul_6 = mul_5 * dy
    add_2 = add_1 + mul_6
    div_4 = sum_4 * INV_HIDDEN_F
    add_3 = add_2 + div_4

    ct.store(add_3_ptr, index=(row, 0), tile=add_3)

    keep = ct.astype(ct.load(keep_ptr, index=(row, 0), shape=(1, BLOCK_H),
                             padding_mode=ct.PaddingMode.ZERO), ct.float32)
    add_3_bf = ct.astype(add_3, ct.bfloat16)
    scaled = keep * DROPOUT_SCALE
    scaled_bf = ct.astype(scaled, ct.bfloat16)
    drop_bf = ct.astype(ct.astype(add_3_bf, ct.float32) *
                        ct.astype(scaled_bf, ct.float32), ct.bfloat16)
    ct.store(view_3_ptr, index=(row, 0), tile=drop_bf)

    # Partial column sums (single-row groups)
    acc_x = ct.where(col_mask, x, zero_2d)
    acc_mul3 = ct.where(col_mask, mul_3, zero_2d)
    acc_drop = ct.where(col_mask,
                        ct.astype(drop_bf, ct.float32), zero_2d)
    ct.store(partials_ptr, index=(row, 0, 0),
             tile=ct.reshape(acc_x, (1, 1, BLOCK_H)))
    ct.store(partials_ptr, index=(row, 1, 0),
             tile=ct.reshape(acc_mul3, (1, 1, BLOCK_H)))
    ct.store(partials_ptr, index=(row, 2, 0),
             tile=ct.reshape(acc_drop, (1, 1, BLOCK_H)))


@ct.kernel
def _finalize_partials_kernel(
    partials_ptr,  # f32 (NUM_GROUPS, 3, BLOCK_H)
    view_1_ptr,    # f32 (HIDDEN,)
    view_2_ptr,    # f32 (HIDDEN,)
    conv_4_ptr,    # f32 (HIDDEN,)
    NUM_GROUPS: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    FINAL_BLOCK_H: ct.Constant[int],
    BLOCK_H_TOTAL: ct.Constant[int],
):
    col_block = ct.bid(0)
    x = ct.load(partials_ptr, index=(0, 0, col_block),
                shape=(NUM_GROUPS, 1, FINAL_BLOCK_H))
    mul3 = ct.load(partials_ptr, index=(0, 1, col_block),
                   shape=(NUM_GROUPS, 1, FINAL_BLOCK_H))
    drop = ct.load(partials_ptr, index=(0, 2, col_block),
                   shape=(NUM_GROUPS, 1, FINAL_BLOCK_H))

    # Mask columns outside HIDDEN
    col_idx = ct.arange(FINAL_BLOCK_H, dtype=ct.int32) + col_block * FINAL_BLOCK_H
    col_valid = col_idx < HIDDEN_

    s_x = ct.reshape(ct.sum(x, axis=0), (FINAL_BLOCK_H,))
    s_mul3 = ct.reshape(ct.sum(mul3, axis=0), (FINAL_BLOCK_H,))
    s_drop = ct.reshape(ct.sum(drop, axis=0), (FINAL_BLOCK_H,))
    s_drop_rounded = ct.astype(ct.astype(s_drop, ct.bfloat16), ct.float32)

    zero_1d = ct.full((FINAL_BLOCK_H,), 0.0, dtype=ct.float32)
    s_x_out = ct.where(col_valid, s_x, zero_1d)
    s_mul3_out = ct.where(col_valid, s_mul3, zero_1d)
    s_drop_out = ct.where(col_valid, s_drop_rounded, zero_1d)

    ct.store(view_1_ptr, index=(col_block,), tile=s_x_out)
    ct.store(view_2_ptr, index=(col_block,), tile=s_mul3_out)
    ct.store(conv_4_ptr, index=(col_block,), tile=s_drop_out)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


@oracle_impl(hardware="B200", point="801fb66e",
             BLOCK_H=1024, FINAL_BLOCK_H=16)
def oracle_forward(inputs, *, BLOCK_H: int, FINAL_BLOCK_H: int):
    (arg0, arg1, arg2, arg3, arg4, arg5,
     shape0, shape1, shape2, shape3, shape4, shape5) = inputs

    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])
    device = arg0.device

    add_3 = torch.empty_strided(
        _shape_tuple(shape3),
        (int(shape0[1]) * hidden, hidden, 1),
        device=device, dtype=torch.float32,
    )
    view_3 = torch.empty_strided(
        _shape_tuple(shape4), (hidden, 1),
        device=device, dtype=torch.bfloat16,
    )
    view_1 = torch.empty_strided(_shape_tuple(shape1), (1,), device=device, dtype=torch.float32)
    view_2 = torch.empty_strided(_shape_tuple(shape2), (1,), device=device, dtype=torch.float32)
    full = torch.tensor(0.0, device=device, dtype=torch.float32)
    convert_element_type_4 = torch.empty_strided(
        _shape_tuple(shape5), (1,), device=device, dtype=torch.float32,
    )

    residual = arg2.view(rows, hidden)
    denom_base = arg3.view(rows)
    prev_grad = arg4.view(rows, hidden)
    keep = arg5.view(rows, hidden)
    add_3_2d = add_3.view(rows, hidden)
    view_3_2d = view_3.view(rows, hidden)

    partials = torch.empty((rows, 3, BLOCK_H), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    inv_hidden = 1.0 / hidden
    ct.launch(
        stream, (rows, 1, 1), _row_partials_kernel,
        (arg0, arg1, residual, denom_base, prev_grad, keep,
         add_3_2d, view_3_2d, partials, hidden, inv_hidden, BLOCK_H),
    )
    num_finalize = (hidden + FINAL_BLOCK_H - 1) // FINAL_BLOCK_H
    ct.launch(
        stream, (num_finalize, 1, 1), _finalize_partials_kernel,
        (partials, view_1, view_2, convert_element_type_4,
         rows, hidden, FINAL_BLOCK_H, BLOCK_H),
    )

    permute = torch.as_strided(view_3, (hidden, rows), (1, hidden))
    return view_1, view_2, full, add_3, view_3, permute, convert_element_type_4
