"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 batchnorm-plus-hard-swish reduction fragment by tiling the channels-last producer once, preserving both explicit bf16 rounding boundaries, writing the returned f32 activation tensor and bf16 masked spatial-sum tensor, and finalizing the sibling per-channel sum from that same bf16 output, whereas Inductor schedules the broadcast pointwise chain, spatial reduction, returned tensor stores, mask epilogue, and final channel reduction as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a full-scope multi-output reduction template that keeps a returned channels-last pointwise producer, a dependent spatial reduction, and the downstream batch-channel reduction in one fused plan while respecting bf16 casts; the fix is SCHEDULER_FUSION: teach Inductor to fuse compatible returned pointwise producers with dependent small-domain reductions and their sibling reduction epilogues under explicit dtype-boundary constraints."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _round_bf16_to_fp32(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _producer_spatial_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    arg7_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    arg1_s0: tl.constexpr,
    arg1_s1: tl.constexpr,
    arg1_s2: tl.constexpr,
    arg1_s3: tl.constexpr,
    arg6_s0: tl.constexpr,
    arg6_s1: tl.constexpr,
    arg6_s2: tl.constexpr,
    arg6_s3: tl.constexpr,
    out1_s0: tl.constexpr,
    out1_s1: tl.constexpr,
    out1_s2: tl.constexpr,
    out1_s3: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)

    hw = tl.arange(0, BLOCK_HW)[:, None]
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    c_vec = c_block * BLOCK_C + tl.arange(0, BLOCK_C)

    hw_mask = hw < H * W
    c_mask = c < C
    elem_mask = hw_mask & c_mask
    c_vec_mask = c_vec < C

    h = hw // W
    w = hw - h * W

    x_offsets = n * arg1_s0 + c * arg1_s1 + h * arg1_s2 + w * arg1_s3
    grad_offsets = n * arg6_s0 + c * arg6_s1 + h * arg6_s2 + w * arg6_s3
    out1_offsets = n * out1_s0 + c * out1_s1 + h * out1_s2 + w * out1_s3

    mean = tl.load(arg2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(arg3_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(arg4_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(arg5_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    x = tl.load(arg1_ptr + x_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    grad = tl.load(arg6_ptr + grad_offsets, mask=elem_mask, other=0.0).to(tl.float32)

    affine = ((x - mean) * invstd) * weight + bias
    affine_bf16 = _round_bf16_to_fp32(affine)
    tl.store(out1_ptr + out1_offsets, affine_bf16, mask=elem_mask)

    shifted = affine_bf16 + 3.0
    clipped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    hswish = (affine_bf16 * clipped) / 6.0
    hswish_bf16 = _round_bf16_to_fp32(hswish)
    product_bf16 = _round_bf16_to_fp32(grad * hswish_bf16)
    spatial_sum = tl.sum(tl.where(elem_mask, product_bf16, 0.0), axis=0)
    spatial_sum_bf16 = _round_bf16_to_fp32(spatial_sum)

    arg0 = tl.load(arg0_ptr + n * C + c_vec, mask=c_vec_mask, other=0.0).to(tl.float32)
    tl.store(out0_ptr + n * C + c_vec, arg0, mask=c_vec_mask)

    scaled = spatial_sum_bf16 * 0.16666666666666666
    fill = tl.load(arg7_ptr).to(tl.float32)
    masked = tl.where((arg0 > -3.0) & (arg0 < 3.0), scaled, fill)
    masked_bf16 = _round_bf16_to_fp32(masked)
    tl.store(out2_ptr + n * C + c_vec, masked_bf16, mask=c_vec_mask)


@triton.jit
def _final_channel_sum_kernel(
    out2_ptr,
    out3_ptr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    n = tl.arange(0, BLOCK_N)[:, None]
    mask = (c < C) & (n < N)
    vals = tl.load(out2_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    channel_sum = tl.sum(vals, axis=0)
    channel_sum_bf16 = _round_bf16_to_fp32(channel_sum)
    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(out3_ptr + c_vec, channel_sum_bf16, mask=c_vec < C)


def _channels_last_stride(n: int, c: int, h: int, w: int):
    return (c * h * w, 1, w * c, c)


# 6425bce8: (N=32, C=480, H=14, W=14)
@oracle_impl(hardware="B200", point="6425bce8", BLOCK_HW=256, BLOCK_C=32, FINAL_BLOCK_C=64, num_warps=8)
# 64645353: (N=32, C=672, H=14, W=14)
@oracle_impl(hardware="B200", point="64645353", BLOCK_HW=256, BLOCK_C=16, FINAL_BLOCK_C=64, num_warps=8)
# e7e0ad11: (N=32, C=672, H=7, W=7)
@oracle_impl(hardware="B200", point="e7e0ad11", BLOCK_HW=64, BLOCK_C=32, FINAL_BLOCK_C=64, num_warps=4)
# 239d1349: (N=32, C=960, H=7, W=7)
@oracle_impl(hardware="B200", point="239d1349", BLOCK_HW=64, BLOCK_C=32, FINAL_BLOCK_C=64, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs

    n = int(arg1_1.shape[0])
    c = int(arg1_1.shape[1])
    h = int(arg1_1.shape[2])
    w = int(arg1_1.shape[3])

    out0 = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        (n, c, h, w),
        _channels_last_stride(n, c, h, w),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out2 = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    out3 = torch.empty_strided(
        (c,),
        (1,),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    _producer_spatial_kernel[(n, triton.cdiv(c, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        out0,
        out1,
        out2,
        arg1_s0=arg1_1.stride(0),
        arg1_s1=arg1_1.stride(1),
        arg1_s2=arg1_1.stride(2),
        arg1_s3=arg1_1.stride(3),
        arg6_s0=arg6_1.stride(0),
        arg6_s1=arg6_1.stride(1),
        arg6_s2=arg6_1.stride(2),
        arg6_s3=arg6_1.stride(3),
        out1_s0=out1.stride(0),
        out1_s1=out1.stride(1),
        out1_s2=out1.stride(2),
        out1_s3=out1.stride(3),
        C=c,
        H=h,
        W=w,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _final_channel_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        out2,
        out3,
        C=c,
        N=n,
        BLOCK_N=32,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=4,
        num_stages=3,
    )

    return out0, out1, out2, out3
