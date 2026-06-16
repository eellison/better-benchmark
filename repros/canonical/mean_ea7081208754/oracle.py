"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 BN-affine, explicit bf16 ReLU rounding, materialized channels-last channel cat, and keepdim spatial mean for both cat halves in one output-owned pass, whereas Inductor lowers the BN/ReLU producer, cat materialization, and following mean as generic scheduled work with avoidable layout traffic and repeated producer indexing; Inductor cannot do this today because its scheduler does not keep a fixed channel cat virtual across a multi-output boundary while also sinking the BN/ReLU producer into the reduction that consumes the cat; the fix is SCHEDULER_FUSION: represent fixed channel-cat materialization and reductions over that cat as one fused schedule with direct multi-source output layout stores."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _cat_bn_relu_mean_kernel(
    mean_ptr,
    conv_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    left_ptr,
    cat_ptr,
    mean_out_ptr,
    TOTAL_ROWS: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    OUT_C: tl.constexpr,
    CONV_S0: tl.constexpr,
    CONV_S1: tl.constexpr,
    CONV_S2: tl.constexpr,
    CONV_S3: tl.constexpr,
    LEFT_S0: tl.constexpr,
    LEFT_S1: tl.constexpr,
    LEFT_S2: tl.constexpr,
    LEFT_S3: tl.constexpr,
    CAT_S0: tl.constexpr,
    CAT_S1: tl.constexpr,
    CAT_S2: tl.constexpr,
    CAT_S3: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    spatial = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    spatial_mask = spatial < HW

    n = rows // OUT_C
    out_c = rows - n * OUT_C
    use_bn = out_c >= C
    c = out_c - tl.where(use_bn, C, 0)
    h = spatial // W
    w = spatial - h * W
    valid = row_mask[:, None] & spatial_mask[None, :]

    left_offsets = (
        n[:, None] * LEFT_S0
        + c[:, None] * LEFT_S1
        + h[None, :] * LEFT_S2
        + w[None, :] * LEFT_S3
    )
    conv_offsets = (
        n[:, None] * CONV_S0
        + c[:, None] * CONV_S1
        + h[None, :] * CONV_S2
        + w[None, :] * CONV_S3
    )
    cat_offsets = (
        n[:, None] * CAT_S0
        + out_c[:, None] * CAT_S1
        + h[None, :] * CAT_S2
        + w[None, :] * CAT_S3
    )

    left = tl.load(left_ptr + left_offsets, mask=valid & (~use_bn[:, None]), other=0.0)

    x = tl.load(conv_ptr + conv_offsets, mask=valid & use_bn[:, None], other=0.0).to(tl.float32)
    running_mean = tl.load(mean_ptr + c, mask=row_mask & use_bn, other=0.0).to(tl.float32)
    running_var = tl.load(var_ptr + c, mask=row_mask & use_bn, other=1.0).to(tl.float32)
    gamma = tl.load(weight_ptr + c, mask=row_mask & use_bn, other=0.0).to(tl.float32)
    beta = tl.load(bias_ptr + c, mask=row_mask & use_bn, other=0.0).to(tl.float32)

    centered = x - running_mean[:, None]
    invstd = 1.0 / libdevice.sqrt(running_var + 1.0e-5)
    y = centered * invstd[:, None]
    y = y * gamma[:, None] + beta[:, None]
    y_bf16 = y.to(tl.bfloat16)
    zero = tl.full((BLOCK_ROWS, BLOCK_HW), 0.0, tl.bfloat16)
    relu = tl.where(y_bf16 < zero, zero, y_bf16)

    cat_values = tl.where(use_bn[:, None], relu, left)
    tl.store(cat_ptr + cat_offsets, cat_values, mask=valid)

    reduced = tl.sum(tl.where(valid, cat_values.to(tl.float32), 0.0), axis=1) * (1.0 / HW)
    tl.store(mean_out_ptr + rows, reduced, mask=row_mask)


def _launch(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int):
    running_mean, conv, running_var, weight, bias, left = inputs
    n = int(conv.shape[0])
    c = int(conv.shape[1])
    h = int(conv.shape[2])
    w = int(conv.shape[3])
    out_c = c * 2
    hw = h * w
    cat = torch.empty_strided(
        (n, out_c, h, w),
        (out_c * hw, 1, out_c * w, out_c),
        device=conv.device,
        dtype=torch.bfloat16,
    )
    mean_out = torch.empty_strided(
        (n, out_c, 1, 1),
        (out_c, 1, 1, 1),
        device=conv.device,
        dtype=torch.bfloat16,
    )
    total_rows = n * out_c
    _cat_bn_relu_mean_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        running_mean,
        conv,
        running_var,
        weight,
        bias,
        left,
        cat,
        mean_out,
        TOTAL_ROWS=total_rows,
        C=c,
        H=h,
        W=w,
        HW=hw,
        OUT_C=out_c,
        CONV_S0=conv.stride(0),
        CONV_S1=conv.stride(1),
        CONV_S2=conv.stride(2),
        CONV_S3=conv.stride(3),
        LEFT_S0=left.stride(0),
        LEFT_S1=left.stride(1),
        LEFT_S2=left.stride(2),
        LEFT_S3=left.stride(3),
        CAT_S0=cat.stride(0),
        CAT_S1=cat.stride(1),
        CAT_S2=cat.stride(2),
        CAT_S3=cat.stride(3),
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return cat, mean_out


@oracle_impl(hardware="B200", point="5339df69", BLOCK_ROWS=8, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="25326742", BLOCK_ROWS=4, BLOCK_HW=256, num_warps=8)
@oracle_impl(hardware="B200", point="7258878f", BLOCK_ROWS=4, BLOCK_HW=256, num_warps=8)
@oracle_impl(hardware="B200", point="a5185a17", BLOCK_ROWS=1, BLOCK_HW=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int):
    return _launch(inputs, BLOCK_ROWS=BLOCK_ROWS, BLOCK_HW=BLOCK_HW, num_warps=num_warps)
