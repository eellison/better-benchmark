"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full ShuffleNet bf16 inference batch-norm affine, explicit bf16 cast, NaN-preserving bf16 ReLU, virtual cat, channel-shuffle clone layout, and final contiguous output view by writing even output channels from the strided skip input and odd output channels from the BN/ReLU branch in one Triton kernel, whereas Inductor currently lowers the BN/ReLU producer, cat, view, permute, clone, and final view as generic pointwise and layout work with intermediate layout traffic; Inductor cannot do this today because scheduler fusion does not push the fixed channel-shuffle consumer indexing back through a mixed virtual-cat producer and per-channel affine epilogue while preserving bf16 rounding boundaries and NaN ReLU behavior; the fix is SCHEDULER_FUSION: teach scheduler/codegen to represent fixed channel-shuffle cats as direct multi-source output-layout stores and sink affine pointwise producers into that final layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-5


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
def _bn_relu_shuffle_kernel(
    mean_ptr,
    conv_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    skip_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    WIDTH: tl.constexpr,
    SKIP_STRIDE_N: tl.constexpr,
    SKIP_STRIDE_C: tl.constexpr,
    SKIP_STRIDE_H: tl.constexpr,
    SKIP_STRIDE_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    safe_offsets = tl.where(mask, offsets, 0)

    spatial = safe_offsets % HW
    channel = (safe_offsets // HW) % CHANNELS
    batch = safe_offsets // (CHANNELS * HW)
    height = spatial // WIDTH
    width = spatial - height * WIDTH

    x = tl.load(conv_ptr + safe_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    centered = _f32_add(x, -mean)
    inv_std = 1.0 / tl.sqrt(_f32_add(var, 1.0e-5))
    inv_std = _f32_mul(inv_std, 1.0)
    normalized = _f32_mul(centered, inv_std)
    scaled = _f32_mul(normalized, weight)
    affine = _f32_add(scaled, bias)
    affine_bf16 = affine.to(tl.bfloat16)
    zero_bf16 = tl.full((BLOCK,), 0.0, tl.float32).to(tl.bfloat16)
    relu = tl.where((affine_bf16 > zero_bf16) | (affine_bf16 != affine_bf16), affine_bf16, zero_bf16)

    skip_offset = (
        batch * SKIP_STRIDE_N
        + channel * SKIP_STRIDE_C
        + height * SKIP_STRIDE_H
        + width * SKIP_STRIDE_W
    )
    skip = tl.load(skip_ptr + skip_offset, mask=mask, other=0.0)

    out_base = batch * (2 * CHANNELS * HW) + channel * (2 * HW) + spatial
    tl.store(out_ptr + out_base, skip, mask=mask)
    tl.store(out_ptr + out_base + HW, relu, mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _launch(
    arg0_1,
    arg1_1,
    arg2_1,
    arg3_1,
    arg4_1,
    arg5_1,
    out,
    *,
    BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    batch = int(arg1_1.shape[0])
    channels = int(arg1_1.shape[1])
    height = int(arg1_1.shape[2])
    width = int(arg1_1.shape[3])
    hw = height * width
    total = batch * channels * hw
    grid = (triton.cdiv(total, BLOCK),)
    _bn_relu_shuffle_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out,
        TOTAL=total,
        CHANNELS=channels,
        HW=hw,
        WIDTH=width,
        SKIP_STRIDE_N=arg5_1.stride(0),
        SKIP_STRIDE_C=arg5_1.stride(1),
        SKIP_STRIDE_H=arg5_1.stride(2),
        SKIP_STRIDE_W=arg5_1.stride(3),
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@oracle_impl(hardware="B200", point="46084af7", BLOCK=256, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="1e6a9948", BLOCK=256, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="ef47f5e1", BLOCK=256, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1 = inputs
    out_shape = _shape_tuple(_shape_param_1)
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    _launch(
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
