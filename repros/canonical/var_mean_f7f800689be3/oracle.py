"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete f32 NFNet weight-standardization scope in one point-specialized Triton row kernel, including the layout-normalizing clone/view from the strided weight, fp32 population Welford `var_mean(..., dim=(0,2), correction=0, keepdim=True)`, eps=1e-5 rsqrt, Inductor's fp32 `gain * 0.02946278254943948` epilogue multiply, final bf16 standardized-weight cast, and the returned f32 invstd and keepdim-mean side outputs, whereas Inductor lowers the clone/view, var_mean, normalization, scale multiply, cast, and final view through generic norm-template and layout schedules; Inductor cannot do this today because the scheduler does not keep a layout-normalizing clone/view producer virtual through the row statistics reduction while also sinking both scalar side-output stores and the dependent full-row bf16 epilogue store; the fix is SCHEDULER_FUSION: extend the NFNet weight-standardization norm template to fuse strided source reads, Welford row statistics, exact scale/output cast boundaries, and direct side-output/final-layout stores."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _weight_standardization_kernel(
    weight_ptr,
    gain_ptr,
    invstd_ptr,
    standardized_ptr,
    mean_ptr,
    inner_size: tl.constexpr,
    in_channels: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_oc: tl.constexpr,
    stride_ic: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kw: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)[None, :]
    mask = offsets < inner_size
    kernel_elems = kernel_h * kernel_w

    ic = offsets // kernel_elems
    k_linear = offsets - ic * kernel_elems
    kh = k_linear // kernel_w
    kw = k_linear - kh * kernel_w
    src_offsets = (
        channel * stride_oc
        + ic * stride_ic
        + kh * stride_kh
        + kw * stride_kw
    )
    dst_offsets = channel * inner_size + offsets

    x = tl.load(weight_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_acc = tl.zeros([1, BLOCK_K], tl.float32)
    m2_acc = tl.zeros([1, BLOCK_K], tl.float32)
    weight_acc = tl.zeros([1, BLOCK_K], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(mask, mean_next, mean_acc)
    m2_acc = tl.where(mask, m2_next, m2_acc)
    weight_acc = tl.where(mask, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    invstd = libdevice.rsqrt((m2 / inner_size) + 1.0e-5)

    one = tl.arange(0, 1)
    tl.store(invstd_ptr + channel + one, invstd, mask=one == 0)
    tl.store(mean_ptr + channel + one, mean, mask=one == 0)

    gain_scaled = _f32_mul(
        tl.load(gain_ptr + channel).to(tl.float32),
        0.02946278254943948,
    )
    centered = _f32_sub(x, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    y = _f32_mul(normalized, gain_scaled)
    tl.store(standardized_ptr + dst_offsets, y.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="93219f63", BLOCK_K=2048, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="9050d2f1", BLOCK_K=2048, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="666447a7", BLOCK_K=2048, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="58fc5e60", BLOCK_K=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="af508913", BLOCK_K=512, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="cfc2822b", BLOCK_K=256, num_warps=1, num_stages=3)
@oracle_impl(hardware="B200", point="015e7a17", BLOCK_K=32, num_warps=1, num_stages=3)
@oracle_impl(hardware="B200", point="97b48510", BLOCK_K=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="8d69498c", BLOCK_K=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int, num_stages: int):
    weight, gain, _view_shape, out_shape = inputs
    channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    kernel_h = int(weight.shape[2])
    kernel_w = int(weight.shape[3])
    inner_size = in_channels * kernel_h * kernel_w
    out_shape = tuple(int(dim) for dim in out_shape)
    out_stride = (inner_size, kernel_h * kernel_w, kernel_w, 1)

    invstd = torch.empty_strided(
        (channels,),
        (1,),
        device=weight.device,
        dtype=torch.float32,
    )
    standardized = torch.empty_strided(
        out_shape,
        out_stride,
        device=weight.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (1, channels, 1),
        (channels, 1, 1),
        device=weight.device,
        dtype=torch.float32,
    )

    _weight_standardization_kernel[(channels,)](
        weight,
        gain,
        invstd,
        standardized,
        mean,
        inner_size=inner_size,
        in_channels=in_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_oc=int(weight.stride(0)),
        stride_ic=int(weight.stride(1)),
        stride_kh=int(weight.stride(2)),
        stride_kw=int(weight.stride(3)),
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return invstd, standardized, mean
