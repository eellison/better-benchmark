"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV2 bf16 inference-BatchNorm affine, explicit bf16 cast, fp32 ReLU6 minmax, second bf16 cast, 7x7 spatial mean, and final contiguous [16,1280] view in one Triton reduction, whereas Inductor lowers the decomposed BN affine, cast, activation, small spatial mean, and view through generic pointwise and reduction scheduling; Inductor cannot do this today because the normalization scheduler does not keep channel-only BN parameters, required bf16 cast boundaries, activation, and the output-layout epilogue in one fixed plan; the fix is SCHEDULER_FUSION: extend inference BatchNorm lowering to fuse bf16 affine, ReLU6, spatial pooling, and view materialization while preserving exact cast and NaN semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bn_relu6_spatial_mean_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    HW: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw = tl.arange(0, BLOCK_HW)
    row_mask = rows < ROWS
    hw_mask = hw < HW

    n = rows // CHANNELS
    c = rows - n * CHANNELS
    h = hw // WIDTH
    w = hw - h * WIDTH
    offsets = (
        n[:, None] * stride_n
        + c[:, None] * stride_c
        + h[None, :] * stride_h
        + w[None, :] * stride_w
    )
    mask = row_mask[:, None] & hw_mask[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    running_mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    running_var = tl.load(var_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

    centered = x - running_mean[:, None]
    denom = tl.sqrt(running_var[:, None] + 1.0e-5)
    invstd = 1.0 / denom
    normalized = centered * invstd
    affine = normalized * weight[:, None] + bias[:, None]
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    relu6 = tl.where(rounded < 0.0, 0.0, rounded)
    relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
    relu6_bf16 = relu6.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    spatial_sum = tl.sum(tl.where(mask, relu6_bf16, 0.0), axis=1)
    spatial_mean = spatial_sum * (1.0 / 49.0)

    tl.store(
        out_ptr + rows,
        spatial_mean.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=row_mask,
    )


@oracle_impl(hardware="B200", point="c92cf8b8", BLOCK_ROWS=32, BLOCK_HW=64, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int, num_stages: int):
    running_mean, x, running_var, weight, bias, view_shape = inputs
    channels = int(x.shape[1])
    hw = int(x.shape[2] * x.shape[3])
    rows = int(x.shape[0] * channels)
    out = torch.empty_strided(
        tuple(int(dim) for dim in view_shape),
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _bn_relu6_spatial_mean_kernel[(triton.cdiv(rows, BLOCK_ROWS),)](
        running_mean,
        x,
        running_var,
        weight,
        bias,
        out,
        stride_n=x.stride(0),
        stride_c=x.stride(1),
        stride_h=x.stride(2),
        stride_w=x.stride(3),
        CHANNELS=channels,
        WIDTH=int(x.shape[3]),
        HW=hw,
        ROWS=rows,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
