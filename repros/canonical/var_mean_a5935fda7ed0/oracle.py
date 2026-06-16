"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Albert bf16-input residual LayerNorm training scope in one shape-specialized Triton row kernel, including the `[4096,4096] -> [8,512,4096]` metadata view, bf16-to-fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, returned normalized f32 tensor, affine scale/bias epilogue, final bf16 `[4096,4096]` view, and sibling `rsqrt / 4096` output, whereas Inductor lowers the residual producer, row statistics, affine epilogue, bf16 cast/view, and saved-scale store through generic normalization-template scheduling; Inductor cannot fuse this exact returned-output envelope today because its fixed-hidden normalization scheduler does not expose a full-scope row plan that keeps the mixed-dtype residual-add tile resident through the normalized side-output store, affine bf16 cast, and inverse-std side store; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline same-layout residual adds and emit the normalized tensor, bf16 affine view, and inverse-std side tensor from one guarded row schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-12


@triton.jit
def _add_layernorm_kernel(
    flat_bf16_ptr,
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

    flat = tl.load(
        flat_bf16_ptr + offsets,
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
    x = flat + residual

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

    centered = x - mean
    invstd = libdevice.rsqrt(variance + EPS_C)
    normalized = centered * invstd

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
    affine = normalized * weight + bias

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(final_bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
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


@oracle_impl(hardware="B200", point="ba44cc6a", BLOCK_H=4096, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)
    final_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    div_shape = view_shape[:-1] + (1,)

    normalized = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
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
        div_shape,
        _contiguous_stride(div_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _add_layernorm_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
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
