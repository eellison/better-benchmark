"""cuTile port of sum_sum_sum_6178014c1a1b: ALBERT multi-gradient reduction with
LN-backward tail.

The Triton reference uses PTX .rn (which is cuTile's default RTNE). Reductions
and bf16 rounds map straight to torch. We use a cuTile kernel for the final
LN-backward bf16 rounding stage.
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
    # 39 argN inputs + shape params
    args = inputs[:39]
    shape_params = inputs[39:]
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
        arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,
        arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29,
        arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38,
    ) = args
    s = shape_params

    # Ten pair reductions and their sibling raw arg reductions:
    # sum_of(mul(argA, argB), [0,1]) + sum_of(argA, [0,1]) for each (A,B) pair.
    def pair_reduce(a, b):
        return (a * b).sum(dim=[0, 1]), a.sum(dim=[0, 1])

    # Column reductions of the 11 bf16 matrices [4096, 4096]:
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

    # Running sums (before tail):
    add_prod = (prod01 + prod34 + prod67 + prod910 + prod1213 +
                prod1516 + prod1819 + prod2122 + prod2425 + prod2728 + prod3031)
    add_src = (src01 + src34 + src67 + src910 + src1213 +
               src1516 + src1819 + src2122 + src2425 + src2728 + src3031)
    add_bf16 = b2 + b5 + b8 + b11 + b14 + b17 + b20 + b23 + b26 + b29 + b32

    # Tail: LN-backward on (arg34 + arg33.view.f32) with gamma=arg35, xhat=arg36, scale=arg38.
    # arg33 shape [4096, 4096] (bf16). s[11] is the view shape for arg33.
    view11 = arg33.view(*(int(x) for x in s[11]))  # (8, 512, 4096)
    add_30 = arg34 + view11.to(torch.float32)
    mul_11 = add_30 * arg35
    mul_12 = mul_11 * 4096
    sum_34 = mul_11.sum(dim=[2], keepdim=True)
    # arg36 is bf16 xhat -> f32
    xhat_f32 = arg36.to(torch.float32)
    sub_ = xhat_f32 - arg37
    mul_13 = sub_ * arg38  # xhat
    mul_14 = mul_11 * mul_13
    sum_35 = mul_14.sum(dim=[2], keepdim=True)
    mul_15 = mul_13 * sum_35
    sub_a = mul_12 - sum_34
    sub_b = sub_a - mul_15
    div = arg38 / 4096
    mul_16 = div * sub_b
    mul_17 = add_30 * mul_13
    sum_36 = mul_17.sum(dim=[0, 1])
    sum_37 = add_30.sum(dim=[0, 1])

    add_prod_final = add_prod + sum_36
    add_src_final = add_src + sum_37

    # mul_16 -> bf16 dense output (via cuTile)
    numel = mul_16.numel()
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
        (mul_16.contiguous().view(numel), conv_bf16, BLOCK),
    )
    convert_element_type_24 = conv_bf16.view(*mul_16.shape)

    # bf16 view [4096, 4096] via view op — s[12] is view shape for output
    view_12 = convert_element_type_24.view(*(int(x) for x in s[12]))
    permute = view_12.permute(1, 0)
    sum_38 = view_12.sum(dim=[0], keepdim=True, dtype=torch.float32).view(-1).to(torch.bfloat16).to(torch.float32)
    add_33 = add_bf16 + sum_38

    return add_prod_final, add_src_final, convert_element_type_24, view_12, permute, add_33


@oracle_impl(
    hardware="B200",
    point="99a4c701",
    MAIN_BLOCK_ROWS=128,
    MAIN_BLOCK_COLS=64,
    TAIL_ROWS_PER_GROUP=4,
    TAIL_BLOCK_R=1,
    TAIL_BLOCK_C=4096,
    FINAL_BLOCK_COLS=8,
)
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
