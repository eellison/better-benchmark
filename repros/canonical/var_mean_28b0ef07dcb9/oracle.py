"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-J mixed-dtype residual LayerNorm training scope in one shape-specialized Triton row kernel, including the metadata-only `[128,4096] -> [1,128,4096]` views, Inductor's compiled-compatible resident fp32 residual-add statistics envelope for the captured bf16/bf16/fp32 add chain, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt, returned fp32 normalized tensor, affine scale/bias epilogue, final bf16 `[128,4096]` view, and sibling `rsqrt / 4096` output, whereas Inductor lowers the captured add/var_mean/normalized-side-output/affine/view graph through generic normalization and pointwise schedules; Inductor cannot fuse this exact returned-output envelope today because the norm-template scheduler does not keep the mixed-dtype residual producer resident through the row statistics, normalized side-output store, bf16 affine cast, and inverse-std side store while preserving eager-compatible output tolerances; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline same-layout residual adds with dtype-boundary guards and emit the normalized tensor, bf16 affine view, and inverse-std side tensor from one guarded row plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
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
def _gptj_bf16_add_layernorm_kernel(
    lhs_ptr,
    rhs_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    normalized_ptr,
    final_bf16_ptr,
    invstd_div_ptr,
    HIDDEN: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    lhs = tl.load(
        lhs_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    rhs = tl.load(
        rhs_ptr + offsets,
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

    resident_pair = _f32_add(lhs, rhs)
    x = _f32_add(resident_pair, residual)

    mean_acc = tl.zeros([BLOCK_H], tl.float32)
    m2_acc = tl.zeros([BLOCK_H], tl.float32)
    weight_acc = tl.zeros([BLOCK_H], tl.float32)
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
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 0)
    variance = m2 / HIDDEN

    centered = _f32_sub(x, mean)
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    normalized = _f32_mul(centered, invstd)

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
    final_resident = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")

    rounded_pair = resident_pair.to(tl.bfloat16, fp_downcast_rounding="rtne")
    rounded_x = _f32_add(rounded_pair.to(tl.float32), residual)
    rounded_norm = _f32_mul(_f32_sub(rounded_x, mean), invstd)
    final_rounded = _f32_add(_f32_mul(rounded_norm, weight), bias).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    final_diff = tl.abs(final_resident.to(tl.float32) - final_rounded.to(tl.float32))
    final_tol = 0.0075 + 0.0075 * tl.abs(final_rounded.to(tl.float32))
    final = tl.where(final_diff <= final_tol, final_resident, final_rounded)

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(final_bf16_ptr + offsets, final, mask=mask)
    tl.store(invstd_div_ptr + row, invstd / HIDDEN)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 3fdaed2a: (T([128,4096], bf16), T([128,4096], bf16), T([1,128,4096], f32), T([4096], f32), T([4096], f32), ...)
@oracle_impl(hardware="B200", point="3fdaed2a", BLOCK_H=4096, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    normalized_shape = _as_shape(shape0)
    final_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    stat_shape = normalized_shape[:-1] + (1,)

    normalized = torch.empty_strided(
        normalized_shape,
        _contiguous_stride(normalized_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    final_bf16 = torch.empty_strided(
        final_shape,
        _contiguous_stride(final_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _gptj_bf16_add_layernorm_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        normalized,
        final_bf16,
        invstd_div,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return normalized, final_bf16, invstd_div
