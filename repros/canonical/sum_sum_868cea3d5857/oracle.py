"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete cropped SiLU-gradient plus batchnorm-backward return tuple by gathering the fixed negative-pad crop directly into shared channel reductions, preserving the captured bf16 round-trips around the sigmoid producer and reusing the finalized summaries for the f32 vector outputs and bf16 tensor epilogue, whereas Inductor lowers the crop, sibling channel reductions, and dependent tensor epilogue as separate generic reduction/pointwise regions; Inductor cannot do this today because scheduler/codegen does not model structured crop producers as direct reduction sources whose finalized channel summaries can feed a dependent materializing epilogue; the fix is SCATTER_REDUCE: add a structured crop-reduce lowering that maps cropped source coordinates directly into shared channel partials and emits the tensor/vector epilogues from the same channel-reduction plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BN_SCALE = 2.4912308673469386e-06


@triton.jit
def _partial_spatial_reduce_kernel(
    crop_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    partials_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    X_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
):
    x_offsets = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)
    n_offsets = tl.arange(0, N_BLOCK)
    total_x: tl.constexpr = C * H * W
    x_mask = x_offsets < total_x

    x = x_offsets[:, None]
    n = n_offsets[None, :]
    c = x % C
    w = (x // C) % W
    h = (x // (C * W)) % H
    mask = x_mask[:, None] & (n < N)

    activation_offset = n * C * H * W + h * W * C + w * C + c
    crop_offset = n * C * IN_H * IN_W + (h + 1) * IN_W * C + (w + 1) * C + c

    activation = tl.load(activation_ptr + activation_offset, mask=mask, other=0.0).to(tl.float32)
    crop = tl.load(crop_ptr + crop_offset, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=x_mask[:, None], other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=x_mask[:, None], other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=x_mask[:, None], other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=x_mask[:, None], other=0.0).to(tl.float32)

    centered = activation - mean
    affine = ((centered * invstd) * weight) + bias
    affine = affine.to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.sigmoid(affine)
    producer = (crop * sigmoid) * ((affine * (1.0 - sigmoid)) + 1.0)
    producer = producer.to(tl.bfloat16).to(tl.float32)
    producer = tl.where(mask, producer, 0.0)

    sum0 = tl.sum(producer, axis=1)
    sum1 = tl.sum(producer * centered, axis=1)
    tl.store(partials_ptr + x_offsets, sum0, mask=x_mask)
    tl.store(
        partials_ptr + total_x + x_offsets,
        sum1,
        mask=x_mask,
    )


@triton.jit
def _finalize_channel_reduce_kernel(
    partials_ptr,
    stats_ptr,
    sum_out_ptr,
    scaled_sum_out_ptr,
    invstd_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HW
    partial_base = offsets * C + channel

    sum0 = tl.sum(tl.load(partials_ptr + partial_base, mask=mask, other=0.0).to(tl.float32), axis=0)
    sum1 = tl.sum(
        tl.load(partials_ptr + C * HW + partial_base, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + channel).to(tl.float32)
    invstd2 = invstd * invstd

    tl.store(sum_out_ptr + channel, sum0)
    tl.store(scaled_sum_out_ptr + channel, sum1 * invstd)
    tl.store(stats_ptr + channel, sum0 * SCALE)
    tl.store(stats_ptr + C + channel, (sum1 * SCALE) * invstd2)


@triton.jit
def _epilogue_kernel(
    crop_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    stats_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL

    c = offsets % C
    w = (offsets // C) % W
    h = (offsets // (C * W)) % H
    n = offsets // (C * H * W)

    crop_offset = n * C * IN_H * IN_W + (h + 1) * IN_W * C + (w + 1) * C + c

    activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    crop = tl.load(crop_ptr + crop_offset, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    mean_term = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
    centered_term = tl.load(stats_ptr + C + c, mask=mask, other=0.0).to(tl.float32)

    centered = activation - mean
    affine = ((centered * invstd) * weight) + bias
    affine = affine.to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.sigmoid(affine)
    producer = (crop * sigmoid) * ((affine * (1.0 - sigmoid)) + 1.0)
    producer = producer.to(tl.bfloat16).to(tl.float32)
    out = ((producer - (centered * centered_term)) - mean_term) * (invstd * weight)
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="dc48d65c", X_BLOCK=32, HW_BLOCK=4096, EPILOGUE_BLOCK=1024)
@oracle_impl(hardware="B200", point="2bf470ea", X_BLOCK=32, HW_BLOCK=256, EPILOGUE_BLOCK=1024)
def oracle_forward(inputs, *, X_BLOCK, HW_BLOCK, EPILOGUE_BLOCK):
    crop, activation, mean, invstd, weight, bias, _shape_param_0 = inputs
    n = activation.shape[0]
    c = activation.shape[1]
    h = activation.shape[2]
    w = activation.shape[3]
    in_h = crop.shape[2]
    in_w = crop.shape[3]
    numel = n * c * h * w
    total_x = c * h * w

    sum_out = torch.empty_like(weight)
    scaled_sum_out = torch.empty_like(weight)
    tensor_out = torch.empty_strided(
        tuple(activation.shape),
        tuple(activation.stride()),
        device=activation.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty((2, total_x), device=activation.device, dtype=torch.float32)
    stats = torch.empty((2, c), device=activation.device, dtype=torch.float32)

    _partial_spatial_reduce_kernel[(triton.cdiv(total_x, X_BLOCK),)](
        crop,
        activation,
        mean,
        invstd,
        weight,
        bias,
        partials,
        N=n,
        C=c,
        H=h,
        W=w,
        IN_H=in_h,
        IN_W=in_w,
        X_BLOCK=X_BLOCK,
        N_BLOCK=n,
        num_warps=8,
    )
    _finalize_channel_reduce_kernel[(c,)](
        partials,
        stats,
        sum_out,
        scaled_sum_out,
        invstd,
        C=c,
        HW=h * w,
        SCALE=BN_SCALE,
        BLOCK=HW_BLOCK,
        num_warps=4,
    )
    _epilogue_kernel[(triton.cdiv(numel, EPILOGUE_BLOCK),)](
        crop,
        activation,
        mean,
        invstd,
        weight,
        bias,
        stats,
        tensor_out,
        N=n,
        C=c,
        H=h,
        W=w,
        IN_H=in_h,
        IN_W=in_w,
        NUMEL=numel,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return sum_out, scaled_sum_out, tensor_out
