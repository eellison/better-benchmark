"""cuTile port of sum_sum_3ab1da5a9e6c: MobileNetV3 hardsigmoid-gated BN-backward.

Torch does the gate producer and the epilogue; cuTile does the two
[0, 2, 3] channel reductions (sum and dot product with centered).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_pow2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


def _make_reduce_kernel(NHW: int, BLOCK: int):
    @ct.kernel
    def _reduce_kernel(x_ptr, s1_ptr, s2_ptr, centered_ptr):
        c = ct.bid(0)
        x = ct.load(x_ptr, index=(c, 0), shape=(1, BLOCK),
                    padding_mode=ct.PaddingMode.ZERO)
        centered = ct.load(centered_ptr, index=(c, 0), shape=(1, BLOCK),
                           padding_mode=ct.PaddingMode.ZERO)
        col_idx = ct.arange(BLOCK, dtype=ct.int32)
        col_mask = ct.reshape(col_idx < NHW, (1, BLOCK))
        zero = ct.full((1, BLOCK), 0.0, dtype=ct.float32)
        x_m = ct.where(col_mask, x, zero)
        c_m = ct.where(col_mask, centered, zero)
        s1 = ct.sum(x_m)
        s2 = ct.sum(x_m * c_m)
        ct.store(s1_ptr, index=(c,), tile=ct.reshape(s1, (1,)))
        ct.store(s2_ptr, index=(c,), tile=ct.reshape(s2, (1,)))

    return _reduce_kernel


@oracle_impl(hardware="B200", point="e664460a")
@oracle_impl(hardware="B200", point="28bbb717")
@oracle_impl(hardware="B200", point="8f42374e")
@oracle_impl(hardware="B200", point="a6881a60")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     _shape_param_0) = inputs
    device = arg0_1.device

    # Gate: hardsigmoid on arg0 (f32 [N,C,1,1]) -> bf16
    add = arg0_1 + 3.0
    clamped = torch.clamp(add, min=0.0, max=6.0)
    gate_bf = (clamped / 6.0).to(torch.bfloat16)
    # mul: arg1 * gate_bf (broadcasted [N,C,1,1] over spatial)
    mul_bf = (arg1_1.float() * gate_bf.float()).to(torch.bfloat16)
    # expand: broadcast arg2 [N, C, 1, 1] to [N, C, H, W]
    N, C, H, W = arg1_1.shape
    HW = int(H * W)
    # Repro literally divides by 784 (i.e. HW = 28*28); this constant does not
    # depend on N. Kept exact for numerical parity.
    div_1_bf = (arg2_1.float() / 784.0).to(torch.bfloat16)
    div_1_expanded = div_1_bf.expand(N, C, H, W).contiguous()
    add_1_bf = (mul_bf.float() + div_1_expanded.float()).to(torch.bfloat16)

    # where: le(arg3, 0) ? arg4 : add_1
    le = arg3_1 <= 0
    where = torch.where(le, arg4_1, add_1_bf)
    source_f32 = where.float()

    # BN-backward reductions
    squeeze_arg5 = arg5_1.view(C).float()
    unsqueeze_2 = squeeze_arg5.view(1, C, 1, 1)
    centered = arg6_1.float() - unsqueeze_2

    NHW = int(N * H * W)
    BLOCK = _next_pow2(NHW)
    kernel = _make_reduce_kernel(NHW, BLOCK)

    src_perm = source_f32.permute(1, 0, 2, 3).contiguous().view(C, NHW)
    cen_perm = centered.permute(1, 0, 2, 3).contiguous().view(C, NHW)
    if BLOCK != NHW:
        src_p = torch.zeros((C, BLOCK), device=device, dtype=torch.float32)
        src_p[:, :NHW].copy_(src_perm)
        cen_p = torch.zeros((C, BLOCK), device=device, dtype=torch.float32)
        cen_p[:, :NHW].copy_(cen_perm)
    else:
        src_p = src_perm
        cen_p = cen_perm
    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((C,), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (C, 1, 1), kernel, (src_p, sum_1, sum_2, cen_p))

    # Repro's 2.4912308673469386e-06 = 1/401408 = 1/(512*784). This is a constant
    # baked into the graph — it doesn't depend on N at runtime.
    scale = 2.4912308673469386e-06
    mul_2 = sum_1 * scale
    unsqueeze_5 = mul_2.view(1, C, 1, 1)
    mul_3 = sum_2 * scale
    squeeze_1 = arg7_1.view(C).float()
    mul_4 = squeeze_1 * squeeze_1
    mul_5 = mul_3 * mul_4
    unsqueeze_8 = mul_5.view(1, C, 1, 1)
    mul_6 = squeeze_1 * arg8_1
    unsqueeze_11 = mul_6.view(1, C, 1, 1)
    mul_7 = centered * unsqueeze_8
    sub_1 = source_f32 - mul_7
    sub_2 = sub_1 - unsqueeze_5
    mul_8 = sub_2 * unsqueeze_11
    mul_9 = sum_2 * squeeze_1
    out_bf = mul_8.to(torch.bfloat16)

    out_stride = tuple(int(s) for s in arg1_1.stride())
    out = torch.empty_strided((N, C, H, W), out_stride, device=device, dtype=torch.bfloat16)
    out.copy_(out_bf)
    return sum_1, mul_9, out
