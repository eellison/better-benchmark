"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 residual-add LayerNorm training scope in one Triton row kernel, including the flat `[4096,4096]` to `[8,512,4096]` view, the observable bf16 residual add, Inductor's fp32 residual-add statistics envelope for population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, returned mean and rsqrt tensors, affine fp32 epilogue, and final bf16 `[4096,4096]` view output, whereas Inductor lowers the residual add, row statistics, affine epilogue, bf16 cast/view, and sibling returned tensors through generic normalization-template scheduling; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization template does not keep the observable bf16 add, mean/rsqrt side outputs, affine f32 output, and final bf16 view resident across one row-statistics pass; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline same-layout residual adds and emit the add, mean, inverse-std, affine, and cast/view side outputs from one guarded row plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-12


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
def _residual_layernorm_multiout_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_ptr,
    bf16_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    flat = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x_fp32 = _f32_add(flat, residual)
    added_bf16 = x_fp32.to(tl.bfloat16)
    x_bf16 = added_bf16.to(tl.float32)

    mean_out = tl.sum(tl.where(mask, x_fp32, 0.0), axis=0) / HIDDEN
    centered_out = _f32_sub(x_fp32, mean_out)
    variance_out = (
        tl.sum(tl.where(mask, _f32_mul(centered_out, centered_out), 0.0), axis=0)
        / HIDDEN
    )
    invstd_out = libdevice.rsqrt(_f32_add(variance_out, EPS_C))

    mean_affine = tl.sum(tl.where(mask, x_bf16, 0.0), axis=0) / HIDDEN
    centered_affine = _f32_sub(x_bf16, mean_affine)
    variance_affine = (
        tl.sum(tl.where(mask, _f32_mul(centered_affine, centered_affine), 0.0), axis=0)
        / HIDDEN
    )
    invstd_affine = libdevice.rsqrt(_f32_add(variance_affine, EPS_C))
    normalized = _f32_mul(centered_affine, invstd_affine)

    weight = tl.load(
        weight_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(add_ptr + offsets, added_bf16, mask=mask)
    tl.store(mean_ptr + row, mean_out)
    tl.store(rsqrt_ptr + row, invstd_out)
    tl.store(affine_ptr + offsets, affine, mask=mask)
    tl.store(bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 30413f1a: (T([4096,4096], bf16), T([8,512,4096], bf16), T([4096], f32), T([4096], f32), S([8,512,4096]), S([4096,4096]))
@oracle_impl(hardware="B200", point="30413f1a", BLOCK_H=4096, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    stat_shape = norm_shape[:-1] + (1,)

    added = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    affine = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    bf16_view = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _residual_layernorm_multiout_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        added,
        mean,
        rsqrt,
        affine,
        bf16_view,
        ROWS=rows,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return added, mean, rsqrt, affine, bf16_view
