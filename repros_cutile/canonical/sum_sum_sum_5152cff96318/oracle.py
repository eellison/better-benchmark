"""cuTile port of sum_sum_sum_5152cff96318: ConvNeXtV2 GRN-backward.

A single cuTile kernel processes one NHWC pixel per block: it computes the
GRN-backward per-channel arithmetic, produces the bf16 output tile (`mul_5`
in the Triton oracle) and an f32 intermediate `mul_6 = permute * ms2` used
downstream. The three channel reductions (sum_3, sum_4, sum_5) are computed
with torch.sum over the NHWC-contiguous tensors after the kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128


@ct.kernel
def _grn_backward_kernel(
    perm_ptr,       # bf16 [P, C]
    arg2_ptr,       # bf16 [P, C]
    arg3_ptr,       # f32  [P]
    arg4_ptr,       # f32  [P]
    weight_ptr,     # f32  [C]
    mul5_ptr,       # bf16 [P, C]  (linear/flat)
    mul6_ptr,       # f32  [P, C]  (linear/flat)
    C: ct.Constant[int],
    C_PADDED: ct.Constant[int],
    GRN_C: ct.Constant[float],
    INV_GRN_C: ct.Constant[float],
):
    pix = ct.bid(0)

    perm = ct.load(
        perm_ptr, index=(pix, 0), shape=(1, C_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    arg2_bf = ct.load(
        arg2_ptr, index=(pix, 0), shape=(1, C_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight = ct.load(
        weight_ptr, index=(0,), shape=(C_PADDED,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    a3 = ct.load(arg3_ptr, index=(pix,), shape=(1,))
    a4 = ct.load(arg4_ptr, index=(pix,), shape=(1,))

    perm_f = ct.astype(perm, ct.float32)
    arg2_f = ct.astype(arg2_bf, ct.float32)
    weight_2d = ct.reshape(weight, (1, C_PADDED))
    a3_2d = ct.reshape(a3, (1, 1))
    a4_2d = ct.reshape(a4, (1, 1))
    a3_full = ct.broadcast_to(a3_2d, (1, C_PADDED))
    a4_full = ct.broadcast_to(a4_2d, (1, C_PADDED))

    cols = ct.arange(C_PADDED, dtype=ct.int32)
    c_valid = cols < C
    c_valid_2d = ct.reshape(c_valid, (1, C_PADDED))

    m = perm_f * weight_2d
    ms2 = (arg2_f - a3_full) * a4_full

    zero_2d = ct.full((1, C_PADDED), 0.0, dtype=ct.float32)
    m_masked = ct.where(c_valid_2d, m, zero_2d)
    ms2_masked = ct.where(c_valid_2d, ms2, zero_2d)

    s1 = ct.sum(m_masked)
    s2 = ct.sum(m_masked * ms2_masked)

    sub2 = m * GRN_C - s1 - ms2 * s2
    mul5 = (a4_full * INV_GRN_C) * sub2  # f32
    mul5_bf = ct.astype(mul5, ct.bfloat16)
    mul6 = perm_f * ms2  # f32

    # Masked scatter over the valid C channels.
    pix_idx = ct.full((C_PADDED,), pix, dtype=ct.int32)
    flat_idx = pix_idx * C + cols
    mul5_bf_1d = ct.reshape(mul5_bf, (C_PADDED,))
    mul6_1d = ct.reshape(mul6, (C_PADDED,))
    ct.scatter(mul5_ptr, (flat_idx,), mul5_bf_1d, mask=c_valid)
    ct.scatter(mul6_ptr, (flat_idx,), mul6_1d, mask=c_valid)


def _next_pow2(x):
    v = 1
    while v < x:
        v <<= 1
    return v


def _launch(inputs, *, C, H, GRN_C):
    arg0, weight, arg2, arg3, arg4 = inputs
    device = arg0.device

    perm_bf = arg0.permute(0, 2, 3, 1).contiguous()  # bf16 [N, H, W, C]
    arg2_c = arg2.contiguous()                       # bf16 [N, H, W, C]
    arg3_c = arg3.contiguous()                       # f32  [N, H, W, 1]
    arg4_c = arg4.contiguous()                       # f32  [N, H, W, 1]

    total_pixels = N * H * H
    c_padded = _next_pow2(C)

    mul5_bf = torch.empty(
        (N, H, H, C), device=device, dtype=torch.bfloat16
    )
    mul6_f = torch.empty(
        (N, H, H, C), device=device, dtype=torch.float32
    )

    perm_flat = perm_bf.view(total_pixels, C)
    arg2_flat = arg2_c.view(total_pixels, C)
    arg3_flat = arg3_c.view(total_pixels)
    arg4_flat = arg4_c.view(total_pixels)
    mul5_flat = mul5_bf.view(-1)
    mul6_flat = mul6_f.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_pixels, 1, 1),
        _grn_backward_kernel,
        (
            perm_flat, arg2_flat, arg3_flat, arg4_flat, weight,
            mul5_flat, mul6_flat,
            C, c_padded, float(GRN_C), 1.0 / float(GRN_C),
        ),
    )

    # Channel reductions over the NHWC-contiguous tensors.
    sum_3 = mul6_f.sum(dim=[0, 1, 2])                    # f32 [C]
    perm_f = arg0.float().permute(0, 2, 3, 1).contiguous()
    sum_4 = perm_f.sum(dim=[0, 1, 2])                    # f32 [C]
    permute_1 = mul5_bf.permute(0, 3, 1, 2)              # bf16 [N, C, H, W] view
    sum_5_bf = torch.sum(permute_1, dim=[0, 2, 3])       # bf16 [C]
    sum_5_f = sum_5_bf.to(torch.float32)                 # f32 [C]

    return sum_3, sum_4, permute_1, sum_5_f


@oracle_impl(
    hardware="B200",
    point="678a07bc",
    C=160,
    H=28,
    GRN_C=160,
    BLOCK_M=16,
    C_BLOCK=256,
    BLOCK_TILES=1024,
    partial_warps=8,
    final_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="b0299cd5",
    C=320,
    H=14,
    GRN_C=160,
    BLOCK_M=8,
    C_BLOCK=512,
    BLOCK_TILES=1024,
    partial_warps=8,
    final_warps=8,
)
def oracle_forward(inputs, **kwargs):
    return _launch(inputs, C=kwargs["C"], H=kwargs["H"], GRN_C=kwargs["GRN_C"])
