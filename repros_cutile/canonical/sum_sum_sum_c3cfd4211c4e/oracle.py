"""cuTile port of sum_sum_sum_c3cfd4211c4e: Albert LN-backward with extra
input branches, mirroring the sibling 6178014c1a1b port with the tail having
three bf16 view inputs (arg33/arg35/arg36 -> add_32 -> LN backward).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_cast_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    """Elementwise f32 -> bf16."""
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(x, ct.bfloat16))


def _forward(inputs, **kwargs):
    args = inputs[:40]
    s = inputs[40:]
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
        arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,
        arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29,
        arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39,
    ) = args

    def pair_reduce(a, b):
        return (a * b).sum(dim=[0, 1]), a.sum(dim=[0, 1])

    def bf16_col_sum(m):
        return m.sum(dim=[0], keepdim=True, dtype=torch.float32).view(-1).to(torch.bfloat16).to(torch.float32)

    prod01, src01 = pair_reduce(arg0, arg1)
    prod34, src34 = pair_reduce(arg3, arg4)
    prod67, src67 = pair_reduce(arg6, arg7)
    prod910, src910 = pair_reduce(arg9, arg10)
    prod1213, src1213 = pair_reduce(arg12, arg13)
    prod1516, src1516 = pair_reduce(arg15, arg16)
    prod1819, src1819 = pair_reduce(arg18, arg19)
    prod2122, src2122 = pair_reduce(arg21, arg22)
    prod2425, src2425 = pair_reduce(arg24, arg25)
    prod2728, src2728 = pair_reduce(arg27, arg28)
    prod3031, src3031 = pair_reduce(arg30, arg31)

    b2 = bf16_col_sum(arg2)
    b5 = bf16_col_sum(arg5)
    b8 = bf16_col_sum(arg8)
    b11 = bf16_col_sum(arg11)
    b14 = bf16_col_sum(arg14)
    b17 = bf16_col_sum(arg17)
    b20 = bf16_col_sum(arg20)
    b23 = bf16_col_sum(arg23)
    b26 = bf16_col_sum(arg26)
    b29 = bf16_col_sum(arg29)
    b32 = bf16_col_sum(arg32)

    add_prod = (prod01 + prod34 + prod67 + prod910 + prod1213 +
                prod1516 + prod1819 + prod2122 + prod2425 + prod2728 + prod3031)
    add_src = (src01 + src34 + src67 + src910 + src1213 +
               src1516 + src1819 + src2122 + src2425 + src2728 + src3031)
    add_bf16 = b2 + b5 + b8 + b11 + b14 + b17 + b20 + b23 + b26 + b29 + b32

    # Tail: three bf16 views (arg33, arg35, arg36) all summed with arg34,
    # then LN-backward with gamma=arg37, xhat=arg38, scale=arg39.
    view11 = arg33.view(*(int(x) for x in s[11]))
    view12 = arg35.view(*(int(x) for x in s[12]))
    view13 = arg36.view(*(int(x) for x in s[13]))
    add_30 = arg34 + view11.to(torch.float32)
    add_31 = add_30 + view12.to(torch.float32)
    add_32 = add_31 + view13.to(torch.float32)

    mul_11 = add_32 * arg37
    mul_12 = mul_11 * 4096
    sum_34 = mul_11.sum(dim=[2], keepdim=True)
    mul_13 = mul_11 * arg38  # arg38 is xhat here
    sum_35 = mul_13.sum(dim=[2], keepdim=True)
    mul_14 = arg38 * sum_35
    sub_a = mul_12 - sum_34
    sub_b = sub_a - mul_14
    mul_15 = arg39 * sub_b  # arg39 is the scale
    mul_16 = add_32 * arg38
    sum_36 = mul_16.sum(dim=[0, 1])
    sum_37 = add_32.sum(dim=[0, 1])

    add_prod_final = add_prod + sum_36
    add_src_final = add_src + sum_37

    # mul_15 -> bf16 dense output (via cuTile)
    numel = mul_15.numel()
    device = arg0.device
    BLOCK = 1024
    while numel % BLOCK != 0 and BLOCK > 1:
        BLOCK //= 2
    conv_bf16 = torch.empty(numel, device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _bf16_cast_kernel,
        (mul_15.contiguous().view(numel), conv_bf16, BLOCK),
    )
    convert_element_type_25 = conv_bf16.view(*mul_15.shape)

    view_14 = convert_element_type_25.view(*(int(x) for x in s[14]))
    permute = view_14.permute(1, 0)
    sum_38 = view_14.sum(dim=[0], keepdim=True, dtype=torch.float32).view(-1).to(torch.bfloat16).to(torch.float32)
    add_35 = add_bf16 + sum_38

    return mul_15, add_prod_final, add_src_final, view_14, permute, add_35


@oracle_impl(hardware="B200", point="b12d2d03")
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
