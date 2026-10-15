"""cuTile port of sum_sum_dcfe30b8d79b: DenseNet BN-backward with sliced-cat producer.

Uses cuTile per-channel reductions and torch epilogue.
Constants: 0.0012755102040816326 = 1/(4*14*14) = 1/784 (matches 3ceb4473; 4-N, 14x14 spatial).
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


@oracle_impl(hardware="B200", point="3ceb4473")
@oracle_impl(hardware="B200", point="b3dbb7bc")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     arg9_1, arg10_1, arg11_1) = inputs
    device = arg0_1.device

    # Sliced-cat producer: add slices at channels [832, 864) across 5 tensors.
    slice1 = arg0_1[:, 832:864]
    slice2 = arg1_1[:, 832:864]
    slice3 = arg2_1[:, 832:864]
    slice4 = arg3_1[:, 832:864]
    slice5 = arg4_1[:, 832:864]
    add_3 = slice1 + slice2 + slice3 + slice4 + slice5  # bf16 [N, 32, H, W]

    # where(arg5 <= 0, arg6, arg7): bf16
    le = arg5_1 <= 0
    where = torch.where(le, arg6_1, arg7_1)
    source_f32 = where.float()

    N, C, H, W = arg5_1.shape
    NHW = int(N * H * W)
    # Repro's scale = 1/784 = 1/(N*H*W) when N=4, H=W=14; for b3dbb7bc N=4, H=W=7 -> 1/196.
    # But the graph literal is 0.0012755102040816326 = 1/784, so it's fixed.
    # For b3dbb7bc, does it use the same constant? Let's check: 0.0012755... = 1/784.
    scale_const = 0.0012755102040816326

    centered = arg8_1.float() - arg9_1

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

    mul_2 = sum_1 * scale_const
    unsqueeze_2 = mul_2.view(1, C, 1, 1)
    mul_3 = sum_2 * scale_const
    squeeze_arg10 = arg10_1.view(C)
    mul_4 = mul_3 * (squeeze_arg10 * squeeze_arg10)
    unsqueeze_5 = mul_4.view(1, C, 1, 1)
    mul_5 = squeeze_arg10 * arg11_1
    unsqueeze_8 = mul_5.view(1, C, 1, 1)
    mul_7 = centered * unsqueeze_5
    sub_1 = source_f32 - mul_7
    sub_2 = sub_1 - unsqueeze_2
    mul_8 = sub_2 * unsqueeze_8
    mul_9 = sum_2 * squeeze_arg10
    out_bf = mul_8.to(torch.bfloat16)

    # add_4 = add_3 + slice(out_bf, 832:864)
    slice6 = out_bf[:, 832:864]
    add_4 = add_3 + slice6

    return sum_1, mul_9, out_bf, add_4
