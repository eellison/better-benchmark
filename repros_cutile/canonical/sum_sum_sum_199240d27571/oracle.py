"""cuTile port of sum_sum_sum_199240d27571: DebertaV2 GELU + LN backward.

cuTile does the per-row reductions AND the column reductions in-kernel,
matching Triton's kernel structure (row partials + finalize).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327


def _next_pow2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


def _make_row_reduce_kernel(HIDDEN: int, BLOCK_H: int):
    @ct.kernel
    def _kernel(mul_ptr, mul5_ptr, sum_1_ptr, sum_2_ptr):
        row = ct.bid(0)
        mul = ct.load(mul_ptr, index=(row, 0), shape=(1, BLOCK_H),
                      padding_mode=ct.PaddingMode.ZERO)
        mul5 = ct.load(mul5_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
        col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
        col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
        zero = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
        mul_m = ct.where(col_mask, mul, zero)
        prod_m = ct.where(col_mask, mul * mul5, zero)
        s1 = ct.sum(mul_m)
        s2 = ct.sum(prod_m)
        ct.store(sum_1_ptr, index=(row,), tile=ct.reshape(s1, (1,)))
        ct.store(sum_2_ptr, index=(row,), tile=ct.reshape(s2, (1,)))

    return _kernel


def _make_col_sum_kernel(ROWS: int, BLOCK_R: int):
    @ct.kernel
    def _kernel(src_ptr, out_ptr):
        col = ct.bid(0)
        x = ct.load(src_ptr, index=(0, col), shape=(BLOCK_R, 1),
                    padding_mode=ct.PaddingMode.ZERO)
        row_idx = ct.arange(BLOCK_R, dtype=ct.int32)
        row_mask = ct.reshape(row_idx < ROWS, (BLOCK_R, 1))
        x_m = ct.where(row_mask, x, 0.0)
        s = ct.sum(x_m, axis=0)
        ct.store(out_ptr, index=(col,), tile=ct.reshape(s, (1,)))

    return _kernel


@oracle_impl(hardware="B200", point="49fb3d4b")
@oracle_impl(hardware="B200", point="e35c96d4")
@oracle_impl(hardware="B200", point="02c9dc6d")
@oracle_impl(hardware="B200", point="37c6392e")
@oracle_impl(hardware="B200", point="248b11c9")
@oracle_impl(hardware="B200", point="138859e5")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
     shape0, shape1, shape2, shape3) = inputs
    device = arg0_1.device
    ROWS = int(arg0_1.shape[0])
    HIDDEN = int(arg0_1.shape[1])
    view_shape = tuple(int(d) for d in shape0)  # e.g. [8,512,1536]
    n_row = view_shape[0]
    n_seq = view_shape[1]
    rows_2d = n_row * n_seq
    BLOCK_H = _next_pow2(HIDDEN)

    ct_arg0 = arg0_1.view(view_shape).float()
    mul_val = ct_arg0 * arg1_1
    # The Repro's graph has a LITERAL 1536 baked in (from the shape at capture)
    # regardless of runtime HIDDEN. Match it exactly.
    mul_1 = mul_val * 1536.0
    convert1 = arg2_1.view(view_shape).float()
    mul_2 = convert1 * 0.5
    mul_3 = convert1 * RSQRT2
    erf = torch.special.erf(mul_3)
    add_val = erf + 1.0
    mul_4 = mul_2 * add_val
    convert2 = mul_4.to(torch.bfloat16)
    convert3 = convert2.float()
    sub_val = convert3 - arg3_1
    mul_5 = sub_val * arg4_1

    mul_2d = mul_val.view(rows_2d, HIDDEN)
    mul5_2d = mul_5.view(rows_2d, HIDDEN)
    if BLOCK_H != HIDDEN:
        mul_p = torch.zeros((rows_2d, BLOCK_H), device=device, dtype=torch.float32)
        mul_p[:, :HIDDEN].copy_(mul_2d)
        mul5_p = torch.zeros((rows_2d, BLOCK_H), device=device, dtype=torch.float32)
        mul5_p[:, :HIDDEN].copy_(mul5_2d)
    else:
        mul_p = mul_2d
        mul5_p = mul5_2d

    sum_1_1d = torch.empty((rows_2d,), device=device, dtype=torch.float32)
    sum_2_1d = torch.empty((rows_2d,), device=device, dtype=torch.float32)
    kernel = _make_row_reduce_kernel(HIDDEN, BLOCK_H)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (rows_2d, 1, 1), kernel, (mul_p, mul5_p, sum_1_1d, sum_2_1d))
    sum_1 = sum_1_1d.view(n_row, n_seq, 1)
    sum_2 = sum_2_1d.view(n_row, n_seq, 1)

    mul_7 = mul_5 * sum_2
    sub_1 = mul_1 - sum_1
    sub_2 = sub_1 - mul_7
    # Same literal-1536 quirk applies to the div.
    div_val = arg4_1 / 1536.0
    mul_8 = div_val * sub_2
    convert4 = mul_8.to(torch.bfloat16)
    convert5 = convert4.float()
    mul_10 = add_val * 0.5
    mul_11 = convert1 * convert1
    mul_12 = mul_11 * -0.5
    exp_val = torch.exp(mul_12)
    mul_13 = exp_val * NORMAL_PDF_SCALE
    mul_14 = convert1 * mul_13
    add_1 = mul_10 + mul_14
    mul_15 = convert5 * add_1
    convert6 = mul_15.to(torch.bfloat16)
    view_2 = convert6.view(ROWS, HIDDEN)

    mul_9 = ct_arg0 * mul_5

    # Column reductions via cuTile (matches Triton's finalize kernel).
    BLOCK_R = _next_pow2(ROWS)
    mul_9_2d = mul_9.view(ROWS, HIDDEN).contiguous()
    ct_arg0_2d = ct_arg0.view(ROWS, HIDDEN).contiguous()
    view_2_f32 = view_2.to(torch.float32).contiguous()
    if BLOCK_R != ROWS:
        pad_mul9 = torch.zeros((BLOCK_R, HIDDEN), device=device, dtype=torch.float32)
        pad_mul9[:ROWS].copy_(mul_9_2d)
        pad_arg0 = torch.zeros((BLOCK_R, HIDDEN), device=device, dtype=torch.float32)
        pad_arg0[:ROWS].copy_(ct_arg0_2d)
        pad_view2 = torch.zeros((BLOCK_R, HIDDEN), device=device, dtype=torch.float32)
        pad_view2[:ROWS].copy_(view_2_f32)
        src_mul9 = pad_mul9
        src_arg0 = pad_arg0
        src_view2 = pad_view2
        col_kernel = _make_col_sum_kernel(ROWS, BLOCK_R)
    else:
        src_mul9 = mul_9_2d
        src_arg0 = ct_arg0_2d
        src_view2 = view_2_f32
        col_kernel = _make_col_sum_kernel(ROWS, ROWS)
    sum_3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_5 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    ct.launch(stream, (HIDDEN, 1, 1), col_kernel, (src_mul9, sum_3))
    ct.launch(stream, (HIDDEN, 1, 1), col_kernel, (src_arg0, sum_4))
    ct.launch(stream, (HIDDEN, 1, 1), col_kernel, (src_view2, sum_5))
    sum_5_bf = sum_5.to(torch.bfloat16).to(torch.float32)

    permute = view_2.permute(1, 0)
    return sum_3, sum_4, view_2, permute, sum_5_bf
