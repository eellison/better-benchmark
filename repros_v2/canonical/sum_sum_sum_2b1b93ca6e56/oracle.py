"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete functorch per-sample BatchNorm-backward fragment by algebraically collapsing the ReLU-le mask to its fused producer predicate, streaming the shared bf16/f32 `where` tensor once per `(batch, group)`, keeping `sum(where)`, `sum(where * branch1)`, and `sum(where * branch2)` in registers, and emitting the two returned bf16 full-tensor epilogues plus four returned f32 vectors, whereas Inductor lowers the captured cast/view/affine/mask/sum/grouped-sum/epilogue graph through many generic reductions and materialized view intermediates; Inductor cannot do this today because its algebraic simplifier and reduction scheduler do not preserve the shared masked producer and spatial summaries across both dependent grouped BN-backward branches and duplicate channel-sum outputs; the fix is ALGEBRAIC_ELIMINATION: recognize this grouped per-sample BN-backward reduction chain and lower it as one fused multi-output grouped-reduction plan with the bf16 output cast boundaries preserved."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _zero_vectors_kernel(
    out_vec2_ptr,
    out_sum_a_ptr,
    out_vec1_ptr,
    out_sum_b_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_C)
    active = offsets < C
    zeros = tl.zeros((BLOCK_C,), dtype=tl.float32)
    tl.store(out_vec2_ptr + offsets, zeros, mask=active)
    tl.store(out_sum_a_ptr + offsets, zeros, mask=active)
    tl.store(out_vec1_ptr + offsets, zeros, mask=active)
    tl.store(out_sum_b_ptr + offsets, zeros, mask=active)


@triton.jit
def _grouped_bn_backward_kernel(
    residual_bf16_ptr,
    residual_f32_ptr,
    branch1_x_ptr,
    branch1_mean_ptr,
    branch1_inv_ptr,
    branch1_weight_ptr,
    branch1_bias_ptr,
    branch2_x_ptr,
    branch2_mean_ptr,
    branch2_inv_ptr,
    branch2_weight_ptr,
    branch2_bias_ptr,
    full_ptr,
    out_vec2_ptr,
    out_sum_a_ptr,
    out_full2_ptr,
    out_vec1_ptr,
    out_sum_b_ptr,
    out_full1_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_K: tl.constexpr,
    HW: tl.constexpr,
    GROUP_SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid - n * GROUPS

    k = tl.arange(0, BLOCK_K)
    hw = tl.arange(0, BLOCK_HW)
    channels = g * GROUP_K + k
    active = (k[:, None] < GROUP_K) & (hw[None, :] < HW) & (n < N)
    channel_mask = k < GROUP_K
    offsets = n * (C * HW) + hw[None, :] * C + channels[:, None]
    group_offset = n * GROUPS + g

    residual = (
        tl.load(residual_f32_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(residual_bf16_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    )

    x1 = tl.load(branch1_x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean1 = tl.load(branch1_mean_ptr + group_offset).to(tl.float32)
    inv1 = tl.load(branch1_inv_ptr + group_offset).to(tl.float32)
    weight1 = tl.load(branch1_weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias1 = tl.load(branch1_bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    centered1 = x1 - mean1
    affine1 = centered1 * inv1 * weight1[:, None] + bias1[:, None]

    x2 = tl.load(branch2_x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean2 = tl.load(branch2_mean_ptr + group_offset).to(tl.float32)
    inv2 = tl.load(branch2_inv_ptr + group_offset).to(tl.float32)
    weight2 = tl.load(branch2_weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias2 = tl.load(branch2_bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    centered2 = x2 - mean2
    affine2 = centered2 * inv2 * weight2[:, None] + bias2[:, None]

    predicate = (affine1 + affine2) <= 0.0
    full_value = tl.load(full_ptr).to(tl.float32)
    where_value = tl.where(predicate, full_value, residual)

    sum_where = tl.sum(tl.where(active, where_value, 0.0), axis=1)
    sum_x1 = tl.sum(tl.where(active, where_value * x1, 0.0), axis=1)
    sum_x2 = tl.sum(tl.where(active, where_value * x2, 0.0), axis=1)

    grouped_where1 = tl.sum(sum_where * weight1, axis=0)
    grouped_x1 = tl.sum(sum_x1 * weight1, axis=0)
    grouped_where2 = tl.sum(sum_where * weight2, axis=0)
    grouped_x2 = tl.sum(sum_x2 * weight2, axis=0)

    m1 = (((grouped_where1 * mean1 - grouped_x1) * inv1) * inv1) * inv1 * GROUP_SCALE
    bias_term1 = (-m1) * mean1 - grouped_where1 * inv1 * GROUP_SCALE
    out1 = where_value * (inv1 * weight1)[:, None] + x1 * m1 + bias_term1

    m2 = (((grouped_where2 * mean2 - grouped_x2) * inv2) * inv2) * inv2 * GROUP_SCALE
    bias_term2 = (-m2) * mean2 - grouped_where2 * inv2 * GROUP_SCALE
    out2 = where_value * (inv2 * weight2)[:, None] + x2 * m2 + bias_term2

    tl.store(
        out_full1_ptr + offsets,
        out1.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )
    tl.store(
        out_full2_ptr + offsets,
        out2.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )

    tl.atomic_add(out_vec1_ptr + channels, (sum_x1 - sum_where * mean1) * inv1, sem="relaxed", mask=channel_mask)
    tl.atomic_add(out_vec2_ptr + channels, (sum_x2 - sum_where * mean2) * inv2, sem="relaxed", mask=channel_mask)
    tl.atomic_add(out_sum_a_ptr + channels, sum_where, sem="relaxed", mask=channel_mask)
    tl.atomic_add(out_sum_b_ptr + channels, sum_where, sem="relaxed", mask=channel_mask)


def _channels_last_stride(batch: int, channels: int, height: int, width: int):
    return (channels * height * width, 1, width * channels, channels)


# 9705fec3: (T([128,128,4,4], bf16, channels_last), T([128,128,4,4], f32, channels_last), ...)
@oracle_impl(hardware="B200", point="9705fec3", BLOCK_K=4, BLOCK_HW=16, BLOCK_ZERO=128, num_warps=2)
# 95a30d45: (T([128,256,2,2], bf16, channels_last), T([128,256,2,2], f32, channels_last), ...)
@oracle_impl(hardware="B200", point="95a30d45", BLOCK_K=8, BLOCK_HW=4, BLOCK_ZERO=256, num_warps=2)
# 49b9f54f: (T([128,512,1,1], bf16), T([128,512,1,1], f32), ...)
@oracle_impl(hardware="B200", point="49b9f54f", BLOCK_K=16, BLOCK_HW=1, BLOCK_ZERO=512, num_warps=1)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_HW: int,
    BLOCK_ZERO: int,
    num_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        *_shape_params,
    ) = inputs

    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    groups = int(arg3_1.shape[1])
    group_k = channels // groups
    hw = height * width

    out_vec2 = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    out_sum_a = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    out_full2 = torch.empty_strided(
        (batch, channels, height, width),
        _channels_last_stride(batch, channels, height, width),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out_vec1 = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    out_sum_b = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    out_full1 = torch.empty_strided(
        (batch, channels, height, width),
        _channels_last_stride(batch, channels, height, width),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _zero_vectors_kernel[(1,)](
        out_vec2,
        out_sum_a,
        out_vec1,
        out_sum_b,
        C=channels,
        BLOCK_C=BLOCK_ZERO,
        num_warps=1,
    )

    _grouped_bn_backward_kernel[(batch * groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        out_vec2,
        out_sum_a,
        out_full2,
        out_vec1,
        out_sum_b,
        out_full1,
        N=batch,
        C=channels,
        GROUPS=groups,
        GROUP_K=group_k,
        HW=hw,
        GROUP_SCALE=0.015625,
        BLOCK_K=BLOCK_K,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
    )

    return out_vec2, out_sum_a, out_full2, out_vec1, out_sum_b, out_full1
