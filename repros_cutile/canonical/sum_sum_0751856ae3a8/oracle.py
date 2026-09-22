"""cuTile port of sum_sum_0751856ae3a8: GhostNet BN-backward fragment.

sliced-add producer -> f32 promote -> two [0,2,3] channel reductions ->
scalar epilogue for output. Uses cuTile per-channel reductions; the affine
epilogue is torch (fp32).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


REDUCE_SCALE = 1.5570192920918366e-07


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


@oracle_impl(hardware="B200", point="ffa43910")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    slice1 = arg0_1[:, :8].contiguous()
    added_bf = (slice1.float() + arg1_1.float()).to(torch.bfloat16)
    source_f32 = added_bf.float()
    centered = arg2_1.float() - arg3_1

    N, C, H, W = added_bf.shape
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

    mul_1 = sum_1 * REDUCE_SCALE
    unsqueeze_2 = mul_1.view(1, C, 1, 1)
    mul_2 = sum_2 * REDUCE_SCALE
    mul_4 = mul_2 * (arg4_1 * arg4_1)
    unsqueeze_5 = mul_4.view(1, C, 1, 1)
    mul_5 = arg4_1 * arg5_1
    unsqueeze_8 = mul_5.view(1, C, 1, 1)
    mul_6 = centered * unsqueeze_5
    sub_1 = source_f32 - mul_6
    sub_2 = sub_1 - unsqueeze_2
    mul_7 = sub_2 * unsqueeze_8
    mul_8 = sum_2 * arg4_1
    out_bf = mul_7.to(torch.bfloat16)

    out_stride = tuple(int(s) for s in arg1_1.stride())
    out = torch.empty_strided((N, C, H, W), out_stride, device=device, dtype=torch.bfloat16)
    out.copy_(out_bf)
    return sum_1, mul_8, out
