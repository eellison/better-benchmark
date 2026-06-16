"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete EfficientNet bf16 cropped SiLU-gradient batchnorm-backward scope, including the returned f32 channel sums, the returned f32 weight-gradient vector, and the returned channels-last bf16 input-gradient tensor, whereas Inductor lowers the crop, bf16 cast boundaries, sibling channel reductions, and dependent full-tensor epilogue as generic materialized regions; Inductor cannot do this today because its scheduler/codegen does not model structured crop producers as scatter-reduce sources that can feed both channel summaries and the dependent epilogue while preserving bf16 rounding and non-contiguous dense strides; the fix is SCATTER_REDUCE: add a structured crop scatter-reduce lowering that maps source coordinates directly into channel partials and shares finalized reductions with the vector and tensor epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _producer_partials_kernel(
    crop_src_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    partials_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    num_hw_blocks: tl.constexpr,
    num_partials: tl.constexpr,
    crop_s0: tl.constexpr,
    crop_s1: tl.constexpr,
    crop_s2: tl.constexpr,
    crop_s3: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    hw_block = tl.program_id(1)
    c_block = tl.program_id(2)

    hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)[:, None]
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_b = c[None, :]

    hw_mask = hw < hw_size
    c_mask = c < channels
    mask = hw_mask & c_mask[None, :]

    h = hw // width
    w = hw - h * width
    x_offsets = n * x_s0 + c_b * x_s1 + h * x_s2 + w * x_s3
    crop_offsets = n * crop_s0 + c_b * crop_s1 + h * crop_s2 + w * crop_s3

    x = tl.load(activation_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    crop = tl.load(crop_src_ptr + crop_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = x - mean[None, :]
    affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
    affine_bf16 = affine.to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.sigmoid(affine_bf16)
    producer = crop * sigmoid * (affine_bf16 * (1.0 - sigmoid) + 1.0)
    producer_bf16_f32 = producer.to(tl.bfloat16).to(tl.float32)

    vals0 = tl.where(mask, producer_bf16_f32, 0.0)
    vals1 = tl.where(mask, producer_bf16_f32 * centered, 0.0)
    sum0 = tl.sum(vals0, axis=0)
    sum1 = tl.sum(vals1, axis=0)

    partial_row = n * num_hw_blocks + hw_block
    partial_offsets = partial_row * channels + c
    tl.store(partials_ptr + partial_offsets, sum0, mask=c_mask)
    tl.store(partials_ptr + num_partials * channels + partial_offsets, sum1, mask=c_mask)


@triton.jit
def _finalize_channels_kernel(
    partials_ptr,
    sum0_out_ptr,
    sum1_ptr,
    weight_grad_out_ptr,
    invstd_ptr,
    channels: tl.constexpr,
    num_partials: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    c = tl.program_id(0)
    p = tl.arange(0, BLOCK_P)
    mask = p < num_partials
    offsets = p * channels + c

    sum0 = tl.sum(tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    sum1 = tl.sum(
        tl.load(partials_ptr + num_partials * channels + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    tl.store(sum0_out_ptr + c, sum0)
    tl.store(sum1_ptr + c, sum1)
    tl.store(weight_grad_out_ptr + c, sum1 * invstd)


@triton.jit
def _bn_backward_epilogue_kernel(
    crop_src_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    sum0_ptr,
    sum1_ptr,
    out_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    crop_s0: tl.constexpr,
    crop_s1: tl.constexpr,
    crop_s2: tl.constexpr,
    crop_s3: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    out_s3: tl.constexpr,
    BN_SCALE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    hw_block = tl.program_id(1)
    c_block = tl.program_id(2)

    hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)[:, None]
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_b = c[None, :]

    hw_mask = hw < hw_size
    c_mask = c < channels
    mask = hw_mask & c_mask[None, :]

    h = hw // width
    w = hw - h * width
    x_offsets = n * x_s0 + c_b * x_s1 + h * x_s2 + w * x_s3
    crop_offsets = n * crop_s0 + c_b * crop_s1 + h * crop_s2 + w * crop_s3
    out_offsets = n * out_s0 + c_b * out_s1 + h * out_s2 + w * out_s3

    x = tl.load(activation_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    crop = tl.load(crop_src_ptr + crop_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum0 = tl.load(sum0_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = x - mean[None, :]
    affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
    affine_bf16 = affine.to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.sigmoid(affine_bf16)
    producer = crop * sigmoid * (affine_bf16 * (1.0 - sigmoid) + 1.0)
    producer_bf16_f32 = producer.to(tl.bfloat16).to(tl.float32)

    centered_term = sum1[None, :] * BN_SCALE * invstd[None, :] * invstd[None, :]
    mean_term = sum0[None, :] * BN_SCALE
    scale = invstd[None, :] * weight[None, :]
    out = (producer_bf16_f32 - centered * centered_term - mean_term) * scale
    tl.store(out_ptr + out_offsets, out.to(tl.bfloat16), mask=mask)


def _oracle_forward_impl(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs

    batch = int(arg1_1.shape[0])
    channels = int(arg1_1.shape[1])
    height = int(arg1_1.shape[2])
    width = int(arg1_1.shape[3])
    hw_size = height * width
    num_hw_blocks = triton.cdiv(hw_size, BLOCK_HW)
    num_partials = batch * num_hw_blocks

    sum0_out = torch.empty_strided((channels,), (1,), device=arg1_1.device, dtype=torch.float32)
    weight_grad_out = torch.empty_strided((channels,), (1,), device=arg1_1.device, dtype=torch.float32)
    full_out = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    sum1 = torch.empty_strided((channels,), (1,), device=arg1_1.device, dtype=torch.float32)
    partials = torch.empty_strided(
        (2, num_partials, channels),
        (num_partials * channels, channels, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    grid = (batch, num_hw_blocks, triton.cdiv(channels, BLOCK_C))
    _producer_partials_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        partials,
        channels=channels,
        width=width,
        hw_size=hw_size,
        num_hw_blocks=num_hw_blocks,
        num_partials=num_partials,
        crop_s0=arg0_1.stride(0),
        crop_s1=arg0_1.stride(1),
        crop_s2=arg0_1.stride(2),
        crop_s3=arg0_1.stride(3),
        x_s0=arg1_1.stride(0),
        x_s1=arg1_1.stride(1),
        x_s2=arg1_1.stride(2),
        x_s3=arg1_1.stride(3),
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=4,
    )
    _finalize_channels_kernel[(channels,)](
        partials,
        sum0_out,
        sum1,
        weight_grad_out,
        arg3_1,
        channels=channels,
        num_partials=num_partials,
        BLOCK_P=FINAL_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _bn_backward_epilogue_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        sum0_out,
        sum1,
        full_out,
        channels=channels,
        width=width,
        hw_size=hw_size,
        crop_s0=arg0_1.stride(0),
        crop_s1=arg0_1.stride(1),
        crop_s2=arg0_1.stride(2),
        crop_s3=arg0_1.stride(3),
        x_s0=arg1_1.stride(0),
        x_s1=arg1_1.stride(1),
        x_s2=arg1_1.stride(2),
        x_s3=arg1_1.stride(3),
        out_s0=full_out.stride(0),
        out_s1=full_out.stride(1),
        out_s2=full_out.stride(2),
        out_s3=full_out.stride(3),
        BN_SCALE=6.228077168367346e-07,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=4,
    )

    return sum0_out, weight_grad_out, full_out


# afb3f0db: EfficientNet-B0 train, bf16 [128,96,113,113] -> [128,96,112,112].
@oracle_impl(hardware="B200", point="afb3f0db", BLOCK_HW=128, BLOCK_C=64, FINAL_BLOCK=16384, num_warps=8)
# 9eed2ad9: EfficientNet-B0 train, bf16 [128,240,29,29] -> [128,240,28,28].
@oracle_impl(hardware="B200", point="9eed2ad9", BLOCK_HW=128, BLOCK_C=64, FINAL_BLOCK=1024, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK: int,
    num_warps: int,
):
    return _oracle_forward_impl(
        inputs,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        FINAL_BLOCK=FINAL_BLOCK,
        num_warps=num_warps,
    )
