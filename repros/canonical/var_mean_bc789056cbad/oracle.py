"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete CycleGAN bf16 instance-normalization plus residual-add scope in one shape-specialized Triton channel kernel, including bf16-to-fp32 statistics over each fixed `[64,64]` plane, population `var_mean(..., dim=[0,2,3], correction=0, keepdim=True)`, eps=1e-5 `rsqrt`, fp32 normalization, the required bf16 rounding before the residual add, fp32 residual add, final bf16 cast, and the contiguous `[1,256,64,64]` output, whereas Inductor lowers the cast/var_mean/sub/rsqrt/mul/cast/add graph through generic normalization and pointwise schedules; Inductor cannot do this exact full-scope floor today because its normalization scheduler does not select a guarded one-program-per-channel CycleGAN instance-norm template that keeps the spatial reduction tile live through the residual-add epilogue while preserving bf16 cast boundaries; the fix is SCHEDULER_FUSION: add a fixed-spatial instance-norm residual template that fuses the reduction, dtype-aware normalization, and residual output store into the same channel-tiled program."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

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
def _instance_norm_residual_bf16_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    channel = tl.program_id(0)
    elems = tl.arange(0, BLOCK_HW)
    mask = (channel < C) & (elems < HW)
    offsets = channel * HW + elems

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_for_sum = tl.where(elems < HW, x, 0.0)
    mean = tl.sum(x_for_sum, axis=0) / HW
    centered = _f32_sub(x, mean)
    centered_for_sum = tl.where(elems < HW, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_sum, centered_for_sum), axis=0) / HW
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))

    normalized = _f32_mul(centered, invstd).to(tl.bfloat16).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    out = _f32_add(residual, normalized).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


# 50d031e1: (T([1,256,64,64], bf16), T([1,256,64,64], bf16))
@oracle_impl(hardware="B200", point="50d031e1", BLOCK_HW=4096, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_HW: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1 = inputs
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width

    out = torch.empty_strided(
        (1, channels, height, width),
        (channels * hw, hw, width, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _instance_norm_residual_bf16_kernel[(channels,)](
        arg0_1,
        arg1_1,
        out,
        C=channels,
        HW=hw,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
