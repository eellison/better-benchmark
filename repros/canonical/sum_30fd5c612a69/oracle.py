"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle materializes the complete NFNet cropped bf16 exact-GELU-derivative product and accumulates bf16-rounded channel-sum partials from the same top-left crop-gather producer pass, whereas Inductor materializes the negative `constant_pad_nd` crop/pointwise tensor and rereads it through a separate generic N/H/W channel reduction; Inductor cannot do this today because its scheduler cannot coordinate a live returned side output, a structured negative-pad crop producer, and split batch-spatial reduction partials while preserving the explicit bf16/fp32 rounding boundaries; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to emit a fused materialize-plus-partial-reduction producer for returned tensors with structured crop gathers and sibling channel reductions."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327


@triton.jit
def _crop_gelu_materialize_partial_kernel(
    crop_ptr,
    rhs_ptr,
    out_ptr,
    partial_ptr,
    sum_ptr,
    crop_s0: tl.constexpr,
    crop_s1: tl.constexpr,
    crop_s2: tl.constexpr,
    crop_s3: tl.constexpr,
    rhs_s0: tl.constexpr,
    rhs_s1: tl.constexpr,
    rhs_s2: tl.constexpr,
    rhs_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    out_s3: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    M: tl.constexpr,
    GAMMA_: tl.constexpr,
    RSQRT2_: tl.constexpr,
    NORMAL_PDF_SCALE_: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STORE_DIRECT_SUM: tl.constexpr,
):
    row_group = tl.program_id(0)
    channel_group = tl.program_id(1)

    rows = row_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (rows < M) & (channels < C)

    hw = H * W
    n = rows // hw
    spatial = rows - n * hw
    h = spatial // W
    w = spatial - h * W

    crop_offsets = (
        n * crop_s0
        + channels * crop_s1
        + h * crop_s2
        + w * crop_s3
    )
    rhs_offsets = (
        n * rhs_s0
        + channels * rhs_s1
        + h * rhs_s2
        + w * rhs_s3
    )
    out_offsets = (
        n * out_s0
        + channels * out_s1
        + h * out_s2
        + w * out_s3
    )

    lhs = tl.load(crop_ptr + crop_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + rhs_offsets, mask=mask, other=0.0).to(tl.float32)

    scaled_lhs = lhs * GAMMA_
    scaled_lhs_for_sum = scaled_lhs.to(tl.bfloat16).to(tl.float32)
    cdf = (libdevice.erf(rhs * RSQRT2_) + 1.0) * 0.5
    pdf_term = rhs * (libdevice.exp((rhs * rhs) * -0.5) * NORMAL_PDF_SCALE_)
    value_bf16 = (scaled_lhs * (cdf + pdf_term)).to(tl.bfloat16)
    value_for_sum_bf16 = (scaled_lhs_for_sum * (cdf + pdf_term)).to(tl.bfloat16)

    tl.store(out_ptr + out_offsets, value_bf16, mask=mask)

    partial = tl.sum(tl.where(mask, value_for_sum_bf16.to(tl.float32), 0.0), axis=0)
    out_channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = out_channels < C
    if STORE_DIRECT_SUM:
        rounded = partial.to(tl.bfloat16).to(tl.float32)
        tl.store(sum_ptr + out_channels, rounded, mask=channel_mask)
    else:
        tl.store(partial_ptr + row_group * C + out_channels, partial, mask=channel_mask)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_group = tl.program_id(0)
    groups = tl.arange(0, GROUP_BLOCK)[:, None]
    channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (groups < NUM_GROUPS) & (channels < C)

    values = tl.load(
        partial_ptr + groups * C + channels,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    out_channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + out_channels, rounded, mask=out_channels < C)


def _oracle_forward_impl(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    crop_source, rhs = inputs
    n = int(rhs.shape[0])
    c = int(rhs.shape[1])
    h = int(rhs.shape[2])
    w = int(rhs.shape[3])
    m = n * h * w

    out = torch.empty_strided(
        tuple(rhs.shape),
        tuple(rhs.stride()),
        device=rhs.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((c,), (1,), device=rhs.device, dtype=torch.float32)

    num_groups = triton.cdiv(m, ROW_BLOCK)
    direct_sum = num_groups == 1
    if direct_sum:
        partial = sum_out
    else:
        partial = torch.empty_strided(
            (num_groups, c),
            (c, 1),
            device=rhs.device,
            dtype=torch.float32,
        )

    _crop_gelu_materialize_partial_kernel[
        (num_groups, triton.cdiv(c, BLOCK_C))
    ](
        crop_source,
        rhs,
        out,
        partial,
        sum_out,
        crop_s0=crop_source.stride(0),
        crop_s1=crop_source.stride(1),
        crop_s2=crop_source.stride(2),
        crop_s3=crop_source.stride(3),
        rhs_s0=rhs.stride(0),
        rhs_s1=rhs.stride(1),
        rhs_s2=rhs.stride(2),
        rhs_s3=rhs.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        out_s3=out.stride(3),
        N=n,
        C=c,
        H=h,
        W=w,
        M=m,
        GAMMA_=GAMMA,
        RSQRT2_=RSQRT2,
        NORMAL_PDF_SCALE_=NORMAL_PDF_SCALE,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_C=BLOCK_C,
        STORE_DIRECT_SUM=direct_sum,
        num_warps=num_warps,
        num_stages=3,
    )

    if not direct_sum:
        group_block = 1 << (num_groups - 1).bit_length()
        _final_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
            partial,
            sum_out,
            C=c,
            NUM_GROUPS=num_groups,
            GROUP_BLOCK=group_block,
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=8,
            num_stages=3,
        )

    return out, sum_out


# c8e8c8cf: (T([128,64,97,97], bf16, stride=(602176,1,6208,64)), T([128,64,96,96], bf16, stride=(589824,1,6144,64)))
@oracle_impl(hardware="B200", point="c8e8c8cf", ROW_BLOCK=128, BLOCK_C=64, FINAL_BLOCK_C=8, num_warps=8)
# b006f82e: (T([128,256,49,49], bf16, stride=(614656,1,12544,256)), T([128,256,48,48], bf16, stride=(589824,1,12288,256)))
@oracle_impl(hardware="B200", point="b006f82e", ROW_BLOCK=128, BLOCK_C=32, FINAL_BLOCK_C=32, num_warps=8)
# b37b1da0: (T([128,768,25,25], bf16, stride=(480000,1,19200,768)), T([128,768,24,24], bf16, stride=(442368,1,18432,768)))
@oracle_impl(hardware="B200", point="b37b1da0", ROW_BLOCK=64, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=8)
# 4f4e0306: (T([128,768,13,13], bf16, stride=(129792,1,9984,768)), T([128,768,12,12], bf16, stride=(110592,1,9216,768)))
@oracle_impl(hardware="B200", point="4f4e0306", ROW_BLOCK=64, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=8)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    return _oracle_forward_impl(
        inputs,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_C=BLOCK_C,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps,
    )
