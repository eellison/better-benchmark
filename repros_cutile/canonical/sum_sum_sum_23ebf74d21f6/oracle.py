"""cuTile port of sum_sum_sum_23ebf74d21f6: BERT LN-backward + dual dropout.

Compute the graph in torch (which is fully portable — arg8_1 and arg9_1
are input dropout masks, not seeded RNG), except for one cuTile kernel
that handles the per-row LN-backward reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 128
HIDDEN = 768
ROWS = BATCH * SEQ
INV_H = 0.002607561929595828  # 2/(2*768)
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _ln_row_reduction_kernel(
    mul_1_ptr,   # f32 [rows, HIDDEN]  (-add_1 * div_1)
    mul_2_ptr,   # f32 [rows, HIDDEN]  (div_2 * weight, i.e., mul_2 in Repro)
    sum_2_ptr,   # f32 [rows]
    sum_4_ptr,   # f32 [rows] = sum(-mul_2, dim=-1) = -sum(mul_2)
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    m1 = ct.load(
        mul_1_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    m2 = ct.load(
        mul_2_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H))
    m1_masked = ct.where(col_mask, m1, 0.0)
    m2_masked = ct.where(col_mask, m2, 0.0)
    sum_2 = ct.sum(m1_masked)
    sum_neg_mul2 = -ct.sum(m2_masked)
    ct.store(sum_2_ptr, index=(row,), tile=ct.reshape(sum_2, (1,)))
    ct.store(sum_4_ptr, index=(row,), tile=ct.reshape(sum_neg_mul2, (1,)))


@ct.kernel
def _col_sum_kernel(
    src_ptr,  # f32 [ROWS, HIDDEN]
    out_ptr,  # f32 [HIDDEN]
    ROWS_: ct.Constant[int],
):
    col = ct.bid(0)
    x = ct.load(src_ptr, index=(0, col), shape=(ROWS_, 1))
    s = ct.sum(x, axis=0)
    ct.store(out_ptr, index=(col,), tile=ct.reshape(s, (1,)))


@oracle_impl(hardware="B200", point="6b5bc342", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, shape0, shape1, shape2, shape3, shape4, shape5,
     shape6, shape7) = inputs
    device = arg0_1.device

    # view + f32 cast + adds
    view = arg0_1.view(BATCH, SEQ, HIDDEN)
    view_1 = arg1_1.view(BATCH, SEQ, HIDDEN)
    view_2 = arg2_1.view(BATCH, SEQ, HIDDEN)
    conv_f = view.to(torch.float32) + view_1.to(torch.float32)
    add_1 = conv_f + view_2.to(torch.float32)  # f32 [B, S, H]

    mul = arg3_1 * arg4_1  # f32 [B, S, H]
    add_2 = arg5_1 + 1e-06  # [B, S, 1]
    div = mul / add_2
    div_1 = div / add_2
    neg = -add_1
    mul_1 = neg * div_1
    div_2 = add_1 / add_2
    mul_2 = div_2 * arg3_1
    mul_3 = div_2 * arg4_1

    # Row reductions via cuTile
    mul_1_2d = mul_1.contiguous().view(ROWS, HIDDEN)
    mul_2_2d = mul_2.contiguous().view(ROWS, HIDDEN)
    sum_2_1d = torch.empty(ROWS, device=device, dtype=torch.float32)
    sum_4_1d = torch.empty(ROWS, device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _ln_row_reduction_kernel,
        (mul_1_2d, mul_2_2d, sum_2_1d, sum_4_1d, HIDDEN, BLOCK_H),
    )
    sum_2 = sum_2_1d.view(BATCH, SEQ, 1)
    sum_4 = sum_4_1d.view(BATCH, SEQ, 1)

    add_3 = arg6_1 + mul_2
    mul_4 = arg5_1 * 2.0
    div_3 = sum_2 / mul_4
    eq_mask = arg5_1 == 0.0
    where = torch.where(eq_mask, arg7_1, div_3)
    mul_5 = where * INV_H
    mul_6 = mul_5 * arg4_1
    add_4 = add_3 + mul_6
    expand = sum_4.expand(BATCH, SEQ, HIDDEN)
    div_4 = expand / 768.0
    add_5 = add_4 + div_4  # f32 [B, S, H] = mul_8

    conv_3 = arg8_1.to(torch.float32) * DROPOUT_SCALE  # f32
    mul_8 = add_5 * conv_3

    conv_4 = mul_8.to(torch.bfloat16)
    conv_5 = arg9_1.to(torch.bfloat16) * DROPOUT_SCALE
    mul_10 = conv_4 * conv_5  # bf16
    view_5 = mul_10.view(ROWS, HIDDEN)
    permute = view_5.permute(1, 0)

    # Column reductions via cuTile (Triton computes these in-kernel).
    add_1_2d = add_1.view(ROWS, HIDDEN).contiguous()
    mul_3_2d = mul_3.view(ROWS, HIDDEN).contiguous()
    view_5_f32 = view_5.to(torch.float32).contiguous()
    sum_1 = torch.empty(HIDDEN, device=device, dtype=torch.float32)
    sum_3 = torch.empty(HIDDEN, device=device, dtype=torch.float32)
    sum_5 = torch.empty(HIDDEN, device=device, dtype=torch.float32)
    ct.launch(stream, (HIDDEN, 1, 1), _col_sum_kernel,
              (add_1_2d, sum_1, ROWS))
    ct.launch(stream, (HIDDEN, 1, 1), _col_sum_kernel,
              (mul_3_2d, sum_3, ROWS))
    ct.launch(stream, (HIDDEN, 1, 1), _col_sum_kernel,
              (view_5_f32, sum_5, ROWS))
    view_3 = sum_1
    view_4 = sum_3
    conv_7 = sum_5.to(torch.bfloat16).to(torch.float32)

    return view_3, view_4, mul_8, view_5, permute, conv_7
