"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full MobileNetV3 bf16 BN-inference affine, Inductor-style fp32 ReLU, final bf16 activation store, and keepdim spatial mean of that stored activation in one channels-last row-reduction; whereas Inductor lowers the decomposed broadcast normalization, returned activation, and reduction through generic fused regions; Inductor cannot do this today because its scheduler lacks a full-scope multi-output BN-affine ReLU spatial-mean lowering that reuses the channel algebra while preserving the fp32 ReLU, final bf16 cast boundary, and output strides; the fix is SCHEDULER_FUSION: fuse the returned activation producer with the spatial mean consumer while preserving libdevice sqrt, eps, casts, and channels-last layout."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


@triton.jit
def _bn_relu_mean_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    relu_out_ptr,
    mean_out_ptr,
    total_rows: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    hw: tl.constexpr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    spatial = tl.arange(0, BLOCK_HW)
    row_mask = rows < total_rows
    spatial_mask = spatial < hw

    n = rows // channels
    c = rows - n * channels
    h = spatial // width
    w = spatial - h * width
    offsets = (
        n[:, None] * x_stride_n
        + c[:, None] * x_stride_c
        + h[None, :] * x_stride_h
        + w[None, :] * x_stride_w
    )
    valid = row_mask[:, None] & spatial_mask[None, :]

    mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=row_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    invstd = 1.0 / libdevice.sqrt(var + eps)

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    y = ((x - mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
    relu = triton_helpers.maximum(tl.full([1], 0, tl.int32), y)
    relu_bf16 = relu.to(tl.bfloat16)
    relu_f32 = relu_bf16.to(tl.float32)
    reduced = tl.sum(tl.where(spatial_mask[None, :], relu_f32, 0.0), axis=1) / hw

    tl.store(relu_out_ptr + offsets, relu_bf16, mask=valid)
    tl.store(mean_out_ptr + rows, reduced.to(tl.bfloat16), mask=row_mask)


# 37cf4567: bf16 channels-last [512, 120, 28, 28].
@oracle_impl(hardware="B200", point="37cf4567", BLOCK_ROWS=1, num_warps=8)
# bddd3dfb: bf16 channels-last [512, 72, 28, 28].
@oracle_impl(hardware="B200", point="bddd3dfb", BLOCK_ROWS=1, num_warps=8)
def oracle_forward(inputs, *, BLOCK_ROWS: int, num_warps: int):
    mean, x, var, weight, bias = inputs
    batch, channels, height, width = x.shape
    hw = height * width
    relu_out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean_out = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _bn_relu_mean_kernel[(triton.cdiv(batch * channels, BLOCK_ROWS),)](
        mean,
        x,
        var,
        weight,
        bias,
        relu_out,
        mean_out,
        total_rows=batch * channels,
        channels=channels,
        height=height,
        width=width,
        hw=hw,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        eps=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=triton.next_power_of_2(hw),
        num_warps=num_warps,
        num_stages=3,
    )
    return relu_out, mean_out
