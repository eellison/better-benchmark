"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SigLIP ViT patch-position LayerNorm training scope in one Triton row kernel, including the channels-contiguous bf16 `[128,768,16,16]` view/permute patch gather, broadcasted fp32 `[1,256,768]` positional add returned as the f32 pre-normalization tensor, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, returned mean and eps=1e-6 rsqrt tensors, affine fp32 epilogue with final bf16 cast, and flattened `[32768,768]` view output, whereas Inductor lowers the static patch gather, broadcast add, row-statistics reduction, affine cast/view, and sibling stat outputs through generic normalization and layout schedules; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden LayerNorm template does not sink channels-last patch-token producers and broadcast positional adds into the row load plan while also emitting the visible pre-norm, mean, rsqrt, and bf16 affine-view outputs; the fix is SCHEDULER_FUSION: teach the normalization scheduler to fold static patch gathers and broadcast residual producers into one fixed-hidden row plan with all sibling side-output stores."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-6


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
def _patch_position_layernorm_kernel(
    patch_ptr,
    pos_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    mean_out_ptr,
    rsqrt_out_ptr,
    final_bf16_ptr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    PATCH_STRIDE_B: tl.constexpr,
    PATCH_STRIDE_T: tl.constexpr,
    PATCH_STRIDE_H: tl.constexpr,
    POS_STRIDE_T: tl.constexpr,
    POS_STRIDE_H: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]

    batch = row_ids[:, None] // TOKENS
    token = row_ids[:, None] - batch * TOKENS
    cols_2d = cols[None, :]
    offsets = row_ids[:, None] * HIDDEN + cols_2d

    patch = tl.load(
        patch_ptr
        + batch * PATCH_STRIDE_B
        + token * PATCH_STRIDE_T
        + cols_2d * PATCH_STRIDE_H,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    pos = tl.load(
        pos_ptr + token * POS_STRIDE_T + cols_2d * POS_STRIDE_H,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    x = _f32_add(patch, pos)
    tl.store(add_out_ptr + offsets, x, mask=mask)

    x_masked = tl.where(mask, x, 0.0)
    mean = _f32_mul(tl.sum(x_masked, axis=1), 1.0 / HIDDEN)
    centered = _f32_sub(x, mean[:, None])
    centered_masked = tl.where(mask, centered, 0.0)
    variance = _f32_mul(
        tl.sum(_f32_mul(centered_masked, centered_masked), axis=1),
        1.0 / HIDDEN,
    )
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))
    normalized = _f32_mul(centered, invstd[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])

    tl.store(mean_out_ptr + row_ids, mean, mask=row_mask)
    tl.store(rsqrt_out_ptr + row_ids, invstd, mask=row_mask)
    tl.store(final_bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="3006dd3d", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)
    final_shape = _as_shape(shape1)
    batch = int(view_shape[0])
    hidden = int(view_shape[1])
    tokens = int(view_shape[2])
    rows = batch * tokens
    add_shape = (batch, tokens, hidden)
    stat_shape = (batch, tokens, 1)

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    mean_out = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt_out = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    final_bf16 = torch.empty_strided(
        final_shape,
        _contiguous_stride(final_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _patch_position_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        mean_out,
        rsqrt_out,
        final_bf16,
        ROWS=rows,
        TOKENS=tokens,
        HIDDEN=hidden,
        PATCH_STRIDE_B=arg0_1.stride(0),
        PATCH_STRIDE_T=arg0_1.stride(3),
        PATCH_STRIDE_H=arg0_1.stride(1),
        POS_STRIDE_T=arg1_1.stride(1),
        POS_STRIDE_H=arg1_1.stride(2),
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, mean_out, rsqrt_out, final_bf16
