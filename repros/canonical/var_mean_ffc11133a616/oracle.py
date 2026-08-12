"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 NFNet weight-standardization scope in one point-specialized Triton row kernel, including the contiguous `[C,N,1,1] -> [1,C,-1]` view, fp32 population `var_mean(..., dim=(0,2), correction=0, keepdim=True)`, eps=1e-5 rsqrt, the required bf16 rounding of `gain * 0.02551551815399144`, final bf16 output cast, and direct contiguous `[C,N,1,1]` output store, whereas Inductor lowers the view, row statistics, scale multiply, cast, and final view through its generic normalization reduction schedule; Inductor cannot do this today because the scheduler does not select a fixed-inner-size NFNet weight-standardization template that keeps the reduction scalars resident for the full-row epilogue while preserving the bf16 rounding boundaries; the fix is SCHEDULER_FUSION: extend the norm-template lowering to fuse reshape-only producers, row statistics, exact scale rounding, and direct final-layout stores for these 1x1 NFNet weights."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
SCALE = 0.02551551815399144


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
def _channel_norm_scale_bf16_kernel(
    weight_ptr,
    gain_ptr,
    out_ptr,
    INNER_SIZE: tl.constexpr,
    EPS_C: tl.constexpr,
    SCALE_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < INNER_SIZE
    offsets = channel * INNER_SIZE + cols

    x = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_for_sum = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=0) / INNER_SIZE

    centered = _f32_sub(x, mean)
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=0) / INNER_SIZE
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))

    gain_f32 = tl.load(gain_ptr + channel).to(tl.float32)
    gain_scaled = _f32_mul(gain_f32, SCALE_C)
    normalized = _f32_mul(centered, invstd)
    out = _f32_mul(normalized, gain_scaled)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# a24dc267: (T([3072,1536,1,1], bf16), T([3072,1,1,1], bf16), ...)
# b321bd29: (T([1536,768,1,1], bf16), T([1536,1,1,1], bf16), ...)
# e5403093: (T([768,1536,1,1], bf16), T([768,1,1,1], bf16), ...)
# 54d002a2: (T([1536,1536,1,1], bf16), T([1536,1,1,1], bf16), ...)
# 1c377997: (T([1536,512,1,1], bf16), T([1536,1,1,1], bf16), ...)
# 067f0896: (T([768,512,1,1], bf16), T([768,1,1,1], bf16), ...)
# 898896cd: (T([512,256,1,1], bf16), T([512,1,1,1], bf16), ...)
# 59475c6e: (T([256,512,1,1], bf16), T([256,1,1,1], bf16), ...)
# 1098a0cd: (T([256,256,1,1], bf16), T([256,1,1,1], bf16), ...)
# 6844a365: (T([256,128,1,1], bf16), T([256,1,1,1], bf16), ...)
# d94b56b7: (T([128,128,1,1], bf16), T([128,1,1,1], bf16), ...)
# e8722254: (T([2304,1536,1,1], bf16), T([2304,1,1,1], bf16), ...)
# d7001ca8: (T([1536,384,1,1], bf16), T([1536,1,1,1], bf16), ...)
# 1adafdac: (T([384,1536,1,1], bf16), T([384,1,1,1], bf16), ...)
# f054831c: (T([384,512,1,1], bf16), T([384,1,1,1], bf16), ...)
# a9bad761: (T([512,128,1,1], bf16), T([512,1,1,1], bf16), ...)
# d4748368: (T([128,512,1,1], bf16), T([128,1,1,1], bf16), ...)
# 320fe2b8: (T([128,256,1,1], bf16), T([128,1,1,1], bf16), ...)
# f8f31da1: (T([256,64,1,1], bf16), T([256,1,1,1], bf16), ...)
# 12c4ea37: (T([64,128,1,1], bf16), T([64,1,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="a24dc267", BLOCK_K=2048, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="b321bd29", BLOCK_K=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="e5403093", BLOCK_K=2048, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="54d002a2", BLOCK_K=2048, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="1c377997", BLOCK_K=512, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="067f0896", BLOCK_K=512, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="898896cd", BLOCK_K=256, num_warps=1, num_stages=3)
@oracle_impl(hardware="B200", point="59475c6e", BLOCK_K=512, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="1098a0cd", BLOCK_K=256, num_warps=1, num_stages=3)
@oracle_impl(hardware="B200", point="6844a365", BLOCK_K=128, num_warps=1, num_stages=3)
@oracle_impl(hardware="B200", point="d94b56b7", BLOCK_K=128, num_warps=1, num_stages=3)
@oracle_impl(hardware="B200", point="e8722254", BLOCK_K=2048, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="d7001ca8", BLOCK_K=512, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="1adafdac", BLOCK_K=2048, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="f054831c", BLOCK_K=512, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="a9bad761", BLOCK_K=128, num_warps=1, num_stages=3)
@oracle_impl(hardware="B200", point="d4748368", BLOCK_K=512, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="320fe2b8", BLOCK_K=256, num_warps=1, num_stages=3)
@oracle_impl(hardware="B200", point="f8f31da1", BLOCK_K=64, num_warps=1, num_stages=3)
@oracle_impl(hardware="B200", point="12c4ea37", BLOCK_K=128, num_warps=1, num_stages=3)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int, num_stages: int):
    weight, gain, _view_shape, out_shape_param = inputs
    channels = int(weight.shape[0])
    inner_size = int(weight.numel() // channels)
    out_shape = _as_shape(out_shape_param)
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=weight.device,
        dtype=torch.bfloat16,
    )
    _channel_norm_scale_bf16_kernel[(channels,)](
        weight,
        gain,
        out,
        INNER_SIZE=inner_size,
        EPS_C=EPS,
        SCALE_C=SCALE,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
