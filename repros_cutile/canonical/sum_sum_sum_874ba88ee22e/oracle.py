"""cuTile port of sum_sum_sum_874ba88ee22e: functorch dp_cifar10 GN-backward tail.

Substantive @ct.kernel computes the pointwise where + mul producer:
    where = mask ? fill : (arg0 + arg1)   (f32)
    mul   = where * arg4                   (f32)

The rest (group reductions, epilogue) uses torch aten (all graph-capturable).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _where_mul_kernel(
    a_ptr,          # bf16 [total]
    b_ptr,          # bf16 [total]
    mask_ptr,       # bool [total]
    fill_ptr,       # f32 [1]
    arg4_ptr,       # bf16 [total]
    where_ptr,      # f32 [total]
    mul_ptr,        # f32 [total]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    a = ct.load(a_ptr, index=(pid,), shape=(BLOCK,))
    b = ct.load(b_ptr, index=(pid,), shape=(BLOCK,))
    mask = ct.load(mask_ptr, index=(pid,), shape=(BLOCK,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    arg4 = ct.load(arg4_ptr, index=(pid,), shape=(BLOCK,))

    add_f = ct.astype(a, ct.float32) + ct.astype(b, ct.float32)
    fill_scalar = ct.reshape(fill, (1,))
    where_f = ct.where(mask, fill_scalar, add_f)
    ct.store(where_ptr, index=(pid,), tile=where_f)

    arg4_f = ct.astype(arg4, ct.float32)
    mul_f = where_f * arg4_f
    ct.store(mul_ptr, index=(pid,), tile=mul_f)


def _run(inputs, *, BLOCK):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10) = inputs
    device = arg0_1.device

    N, C, H, W = arg0_1.shape
    total = N * C * H * W
    HW = H * W

    a_flat = arg0_1.contiguous().view(total)
    b_flat = arg1_1.contiguous().view(total)
    mask_flat = arg2_1.contiguous().view(total)
    arg4_flat = arg4_1.contiguous().view(total)
    fill_1d = arg3_1.reshape(1)

    where_flat = torch.empty(total, device=device, dtype=torch.float32)
    mul_flat = torch.empty(total, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _where_mul_kernel,
        (a_flat, b_flat, mask_flat, fill_1d, arg4_flat, where_flat, mul_flat, BLOCK),
    )

    where = where_flat.view(N, C, H, W)
    mul = mul_flat.view(N, C, H, W)

    # ---- Reductions via torch (matching eager Repro) ----
    view = mul.view(*tuple(int(d) for d in s0))          # [N, C, HW]
    sum_1 = view.sum(dim=[2])                             # [N, C]
    view_1 = where.view(*tuple(int(d) for d in s1))
    sum_2 = view_1.sum(dim=[2])
    unsq = arg5_1.unsqueeze(0)                            # [1, C]
    mul_1 = sum_1 * unsq
    view_2 = mul_1.view(*tuple(int(d) for d in s2))
    sum_3 = view_2.sum(dim=[2])                           # [N, 32]
    mul_2 = sum_2 * unsq
    view_3 = mul_2.view(*tuple(int(d) for d in s3))
    sum_4 = view_3.sum(dim=[2])                           # [N, 32]
    unsq_1 = arg6_1.unsqueeze(-1)                         # [N, 32, 1]
    view_4 = arg5_1.view(*tuple(int(d) for d in s4))     # [1, 32, 2]
    mul_3 = unsq_1 * view_4                               # [N, 32, 2]

    mul_4 = sum_4 * arg7_1
    sub = mul_4 - sum_3
    mul_5 = sub * arg6_1
    mul_6 = mul_5 * arg6_1
    mul_7 = mul_6 * arg6_1
    mul_8 = mul_7 * 0.0078125
    neg = -mul_8
    mul_9 = neg * arg7_1
    mul_10 = sum_4 * arg6_1
    mul_11 = mul_10 * 0.0078125
    sub_1 = mul_9 - mul_11
    unsq_2 = mul_3.unsqueeze(-1)                          # [N, 32, 2, 1]
    unsq_3 = mul_8.unsqueeze(-1)
    unsq_4 = unsq_3.unsqueeze(-1)                          # [N, 32, 1, 1]
    unsq_5 = sub_1.unsqueeze(-1)
    unsq_6 = unsq_5.unsqueeze(-1)

    view_5 = where.view(*tuple(int(d) for d in s5))       # [N, 32, groups, K]
    mul_12 = view_5 * unsq_2
    view_6 = arg4_1.to(torch.float32).view(*tuple(int(d) for d in s6))
    mul_13 = view_6 * unsq_4
    add_1 = mul_12 + mul_13
    add_2 = add_1 + unsq_6
    view_7 = add_2.view(*tuple(int(d) for d in s7))

    view_8 = sum_1.view(*tuple(int(d) for d in s8))       # [N, 32, 2]
    view_9 = sum_2.view(*tuple(int(d) for d in s9))
    unsq_7 = arg7_1.unsqueeze(-1)
    mul_14 = view_9 * unsq_7
    sub_2 = view_8 - mul_14
    mul_15 = sub_2 * unsq_1
    sum_5 = mul_15.sum(dim=[0])                            # [32, 2]
    view_10 = sum_5.view(*tuple(int(d) for d in s10))     # [64]
    sum_6 = sum_2.sum(dim=[0])                             # [C]

    out_bf = view_7.to(torch.bfloat16)
    # apply strided layout matching arg0
    out_shape = (N, C, H, W)
    out_stride = (C * HW, 1, C * W, C)
    out = torch.empty_strided(out_shape, out_stride, device=device, dtype=torch.bfloat16)
    out.copy_(out_bf)
    return where, view_10, sum_6, out


@oracle_impl(hardware="B200", point="44cc7dfb", BLOCK=1024)
@oracle_impl(hardware="B200", point="520f98fe", BLOCK=1024)
@oracle_impl(hardware="B200", point="4ca01415", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    return _run(inputs, BLOCK=BLOCK)
