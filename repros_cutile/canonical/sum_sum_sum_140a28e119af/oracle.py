"""cuTile port of sum_sum_sum_140a28e119af: MegatronBert LN-backward + embedding scatter.

Uses one cuTile kernel for the LN-backward per-row pointwise op (produces mul_7).
Torch handles the outer reductions (sum_3, sum_4) and scatter accumulate.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _ln_bwd_kernel(
    add1_ptr, weight_ptr, xhat_ptr, invstd_ptr, residual_ptr, mask_ptr,
    out_mul7_ptr,
    H: ct.Constant[int],
):
    row = ct.bid(0)
    add_1 = ct.load(add1_ptr, index=(row, 0), shape=(1, H))
    weight = ct.load(weight_ptr, index=(0,), shape=(H,))
    xhat = ct.load(xhat_ptr, index=(row, 0), shape=(1, H))
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, H))
    mask = ct.load(mask_ptr, index=(row, 0), shape=(1, H))

    weight_2d = ct.reshape(weight, (1, H))
    mul = add_1 * weight_2d
    mul_1 = mul * H
    sum_1 = ct.sum(mul, axis=1, keepdims=True)
    mul_2 = mul * xhat
    sum_2 = ct.sum(mul_2, axis=1, keepdims=True)
    mul_3 = xhat * sum_2
    sub = mul_1 - sum_1
    sub_1 = sub - mul_3
    invstd_2d = ct.reshape(invstd, (1, 1))
    mul_4 = invstd_2d * sub_1
    add_2 = residual + mul_4

    mask_f = ct.astype(mask, ct.float32) * 1.1111111111111112
    mul_7 = add_2 * mask_f
    ct.store(out_mul7_ptr, index=(row, 0), tile=mul_7)


@ct.kernel
def _col_sum_kernel_full(
    src_ptr,  # f32 [ROWS, H]
    out_ptr,  # f32 [H]
    ROWS_: ct.Constant[int],
):
    col = ct.bid(0)
    x = ct.load(src_ptr, index=(0, col), shape=(ROWS_, 1))
    s = ct.sum(x, axis=0)
    ct.store(out_ptr, index=(col,), tile=ct.reshape(s, (1,)))


@ct.kernel
def _col_sum_kernel_seq(
    src_ptr,  # f32 [N, S, H]
    out_ptr,  # f32 [S, H]
    N_: ct.Constant[int],
):
    s_idx = ct.bid(0)
    col = ct.bid(1)
    x = ct.load(src_ptr, index=(0, s_idx, col), shape=(N_, 1, 1))
    s = ct.sum(x, axis=0)
    ct.store(out_ptr, index=(s_idx, col), tile=ct.reshape(s, (1, 1)))


@oracle_impl(hardware="B200", point="8ab09cfc")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     arg9_1, arg10_1, arg11_1,
     shape0, shape1, shape2, shape3, shape4, shape5, shape6) = inputs
    device = arg0_1.device

    add_1 = arg1_1.float() + arg2_1.float() + arg3_1.float()  # [8192, 1024]

    H = 1024
    N = 16
    S = 512
    rows = N * S

    xhat_2d = arg5_1.reshape(rows, H)
    invstd_1d = arg6_1.reshape(rows)
    residual_2d = arg7_1.reshape(rows, H)
    mask_2d = arg8_1.reshape(rows, H)

    mul7_2d = torch.empty(rows, H, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _ln_bwd_kernel,
        (add_1, arg4_1, xhat_2d, invstd_1d, residual_2d, mask_2d,
         mul7_2d, H),
    )
    mul_7 = mul7_2d.reshape(N, S, H)

    # Reductions
    add_1_view = add_1.view(N, S, H)
    mul_5 = add_1_view * arg5_1

    # Column reductions via cuTile (Triton computes these in-kernel).
    mul_5_2d = mul_5.view(rows, H).contiguous()
    add_1_2d = add_1.view(rows, H).contiguous()
    sum_3 = torch.empty((H,), device=device, dtype=torch.float32)
    sum_4 = torch.empty((H,), device=device, dtype=torch.float32)
    ct.launch(stream, (H, 1, 1), _col_sum_kernel_full, (mul_5_2d, sum_3, rows))
    ct.launch(stream, (H, 1, 1), _col_sum_kernel_full, (add_1_2d, sum_4, rows))

    # sum_5 = sum over dim 0 -> [1, S, H]. Use cuTile.
    sum_5_flat = torch.empty((S, H), device=device, dtype=torch.float32)
    ct.launch(stream, (S, H, 1), _col_sum_kernel_seq, (mul_7, sum_5_flat, N))
    sum_5 = sum_5_flat.view(1, S, H)

    # Masked scatter into [512, 1024]
    full_a = torch.zeros(tuple(int(d) for d in shape3), device=device,
                          dtype=torch.float32)
    ge = arg9_1 >= 0
    lt = arg9_1 < 512
    ne = arg9_1 != -1
    mask_a = (ge & lt & ne).view(1, S, 1)
    indices_a = arg9_1.view(-1).clamp_min(0).clamp_max(511)  # [512]
    vals_a = torch.where(mask_a.expand_as(sum_5), sum_5,
                         torch.zeros_like(sum_5)).view(S, H)
    full_a.index_add_(0, indices_a, vals_a)

    # Masked scatter into [2, 1024]. full_1 is all True mask.
    full_b = torch.zeros(tuple(int(d) for d in shape5), device=device,
                          dtype=torch.float32)
    indices_b = arg10_1.view(-1).clamp_min(0).clamp_max(1)
    full_b.index_add_(0, indices_b, mul_7.reshape(rows, H))

    # Masked scatter into [29056, 1024] with mul_7
    full_c = torch.zeros(tuple(int(d) for d in shape6), device=device,
                          dtype=torch.float32)
    ge_1 = arg11_1 >= 0
    lt_1 = arg11_1 < 29056
    ne_1 = arg11_1 != 0
    mask_c = (ge_1 & lt_1 & ne_1).view(rows, 1)
    indices_c = arg11_1.view(-1).clamp_min(0).clamp_max(29055)
    vals_c = torch.where(mask_c.expand_as(mul_7.view(rows, H)),
                         mul_7.view(rows, H),
                         torch.zeros_like(mul_7.view(rows, H)))
    full_c.index_add_(0, indices_c, vals_c)

    add_3 = arg0_1.float() + full_c

    return sum_3, sum_4, full_a, full_b, add_3
