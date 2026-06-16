"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 inference-BatchNorm affine, explicit bf16 round trip, SiLU as `x / (exp(-x) + 1)`, second bf16 rounding, spatial mean, captured `as_strided`, and final contiguous `[N,C]` view in one channel-row Triton reduction for both MobileViT and EfficientNet points, whereas Inductor lowers the broadcast affine, activation, explicit casts, small spatial reduction, and view epilogue through a generic fused reduction schedule over channels-last strides; Inductor cannot do this today because its scheduler/codegen lacks a reusable guarded BN-affine plus activation plus fixed-spatial-mean template that reuses per-channel parameters inside the reduction while preserving the captured bf16 cast boundaries and final view layout; the fix is SCHEDULER_FUSION: add a benchmark-gated BN-affine SiLU spatial-mean reduction schedule for dense NCHW tensors with channels-last strides and exact dtype/output semantics."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _invstd_kernel(
    var_ptr,
    invstd_ptr,
    channels: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c < channels
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    var_eps = _f32_add(var, eps)
    sqrt = tl.sqrt_rn(var_eps)
    recip = _f32_div(1.0, sqrt)
    invstd = _f32_mul(recip, 1.0)
    tl.store(invstd_ptr + c, invstd, mask=mask)


@triton.jit
def _bn_silu_spatial_mean_kernel(
    mean_ptr,
    x_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    total_rows: tl.constexpr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw: tl.constexpr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw_offsets = tl.arange(0, BLOCK_HW)
    row_mask = rows < total_rows
    hw_mask = hw_offsets < hw

    n_offsets = rows // channels
    c_offsets = rows - n_offsets * channels
    h_offsets = hw_offsets // width
    w_offsets = hw_offsets - h_offsets * width
    mask = row_mask[:, None] & hw_mask[None, :]

    x_offsets = (
        n_offsets[:, None] * x_stride_n
        + c_offsets[:, None] * x_stride_c
        + h_offsets[None, :] * x_stride_h
        + w_offsets[None, :] * x_stride_w
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[:, None]), bias[:, None])
    rounded = affine.to(tl.bfloat16).to(tl.float32)

    silu = _f32_div(rounded, _f32_add(libdevice.exp(-rounded), 1.0))
    silu_bf16 = silu.to(tl.bfloat16)
    silu_f32 = tl.where(hw_mask[None, :], silu_bf16.to(tl.float32), 0.0)
    reduced = _f32_div(tl.sum(silu_f32, axis=1), hw + 0.0)

    tl.store(out_ptr + rows, reduced.to(tl.bfloat16), mask=row_mask)


def _launch(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int):
    mean, x, var, weight, bias, _shape0, _shape1, shape2 = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    out_shape = tuple(int(dim) for dim in shape2)
    output = torch.empty_strided(
        out_shape,
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    invstd = torch.empty_strided(
        (channels,),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )
    xs = tuple(int(stride) for stride in x.stride())
    _invstd_kernel[(triton.cdiv(channels, 256),)](
        var,
        invstd,
        channels=channels,
        eps=EPS,
        BLOCK_C=256,
        num_warps=4,
        num_stages=3,
    )
    _bn_silu_spatial_mean_kernel[(triton.cdiv(batch * channels, BLOCK_ROWS),)](
        mean,
        x,
        invstd,
        weight,
        bias,
        output,
        total_rows=batch * channels,
        channels=channels,
        width=width,
        hw=hw,
        x_stride_n=xs[0],
        x_stride_c=xs[1],
        x_stride_h=xs[2],
        x_stride_w=xs[3],
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return output


# eaf2d1dd: (T([640], bf16), T([128,640,8,8], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="eaf2d1dd", BLOCK_ROWS=16, BLOCK_HW=64, num_warps=2)
# af408b42: (T([1280], bf16), T([128,1280,7,7], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="af408b42", BLOCK_ROWS=32, BLOCK_HW=64, num_warps=2)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int):
    return _launch(inputs, BLOCK_ROWS=BLOCK_ROWS, BLOCK_HW=BLOCK_HW, num_warps=num_warps)
