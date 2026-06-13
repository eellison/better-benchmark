"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 BatchNorm-affine, explicit bf16 rounding, ReLU side output, bf16 upstream-gradient product, spatial sum, hard-sigmoid gate fallback, visible f32 gate input copy, visible bf16 gated tensor, and final bf16 channel-sum-to-f32 output, whereas Inductor lowers the captured cast/relu/mul/sum/cast/where/cast/sum graph through separate generic pointwise and reduction scheduler fragments; Inductor cannot do this today because its scheduler does not keep the channels-last pointwise producer, spatial reduction, gate epilogue, visible side outputs, and final channel reduction in one full-scope plan while preserving the bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse channels-last BN-affine/ReLU producers with dependent spatial reductions, gate epilogues, and observable side-output materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _relu_spatial_gate_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    grad_ptr,
    gate_ptr,
    full_ptr,
    relu_out_ptr,
    gate_f32_out_ptr,
    gate_bf16_out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.program_id(1)
    hw_offsets = tl.arange(0, BLOCK_HW)
    mask = (c_offsets[None, :] < C) & (hw_offsets[:, None] < HW) & (n < TOTAL_N)
    elem_offsets = n * (C * HW) + hw_offsets[:, None] * C + c_offsets[None, :]

    x = tl.load(x_ptr + elem_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[None, :])
    normalized = _f32_mul(centered, invstd[None, :])
    weighted = _f32_mul(normalized, weight[None, :])
    affine = _f32_add(weighted, bias[None, :])
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    relu = tl.where(affine_bf16 < 0.0, 0.0, affine_bf16)
    relu_bf16 = relu.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(relu_out_ptr + elem_offsets, relu_bf16, mask=mask)

    grad = tl.load(grad_ptr + elem_offsets, mask=mask, other=0.0).to(tl.float32)
    product = _f32_mul(grad, relu_bf16.to(tl.float32))
    product_bf16 = product.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    spatial_sum = tl.sum(tl.where(mask, product_bf16, 0.0), axis=0)

    gate_offsets = n * C + c_offsets
    gate = tl.load(gate_ptr + gate_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    tl.store(gate_f32_out_ptr + gate_offsets, gate, mask=c_offsets < C)

    sum_bf16 = spatial_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    scaled = _f32_mul(sum_bf16, 0.16666666666666666)
    full = tl.load(full_ptr).to(tl.float32)
    in_range = (gate > -3.0) & (gate < 3.0)
    gate_value = tl.where(in_range, scaled, full)
    tl.store(
        gate_bf16_out_ptr + gate_offsets,
        gate_value.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=c_offsets < C,
    )


@triton.jit
def _final_channel_sum_kernel(
    gate_bf16_ptr,
    out_ptr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    c = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)
    active = n_offsets < N
    values = tl.load(gate_bf16_ptr + n_offsets * C + c, mask=active, other=0.0).to(tl.float32)
    channel_sum = tl.sum(values, axis=0)
    rounded = channel_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_ptr + c, rounded)


def _channels_last_stride(batch: int, channels: int, height: int, width: int):
    return (channels * height * width, 1, width * channels, channels)


# 0e39883f: (T([512,72,28,28], bf16, channels_last), ...)
@oracle_impl(hardware="B200", point="0e39883f", BLOCK_HW=1024, BLOCK_C=16, BLOCK_N=512, num_warps_pointwise=8, num_warps_reduce=8)
# ee700829: (T([512,120,28,28], bf16, channels_last), ...)
@oracle_impl(hardware="B200", point="ee700829", BLOCK_HW=1024, BLOCK_C=16, BLOCK_N=512, num_warps_pointwise=8, num_warps_reduce=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    BLOCK_N: int,
    num_warps_pointwise: int,
    num_warps_reduce: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width

    relu_out = torch.empty_strided(
        (batch, channels, height, width),
        _channels_last_stride(batch, channels, height, width),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    gate_f32 = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    gate_bf16 = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    channel_sum = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)

    _relu_spatial_gate_kernel[(triton.cdiv(channels, BLOCK_C), batch)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        relu_out,
        gate_f32,
        gate_bf16,
        C=channels,
        HW=hw,
        TOTAL_N=batch,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_pointwise,
    )

    _final_channel_sum_kernel[(channels,)](
        gate_bf16,
        channel_sum,
        C=channels,
        N=batch,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps_reduce,
    )

    return relu_out, gate_f32, gate_bf16, channel_sum
