"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DeiT bf16/fp32 residual LayerNorm training scope in one Triton row kernel, including the `[25216,192] -> [128,197,192]` metadata view, fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt, returned normalized fp32 tensor for every token row, affine scale/bias only for the live `select(dim=1, index=0).clone()` class-token rows, final bf16 class-token cast, and `rsqrt / 192` side output, whereas Inductor schedules the row normalization and affine epilogue through generic full-token regions before the trailing fixed-token select; Inductor cannot do this today because its normalization scheduler does not split a downstream constant token select through the affine/cast epilogue while preserving all-row normalized and inverse-std side outputs; the fix is ALGEBRAIC_ELIMINATION: push fixed-token selects through row-local affine LayerNorm epilogues so required all-row statistics/normalized outputs remain live but dead non-class-token affine/cast work is eliminated."""

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
def _layernorm_selected_affine_kernel(
    flat_bf16_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    norm_out_ptr,
    class_bf16_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_ids[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    valid = (row_ids[:, None] < ROWS) & (cols < HIDDEN)
    col_valid = cols < HIDDEN
    offsets = rows * HIDDEN + cols

    flat = tl.load(
        flat_bf16_ptr + offsets,
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(residual, flat)

    mean = tl.sum(tl.where(valid, x, 0.0), axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(valid, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    normalized = _f32_mul(centered, invstd[:, None])

    tl.store(norm_out_ptr + offsets, normalized, mask=valid)
    tl.store(div_ptr + row_ids, invstd / HIDDEN, mask=row_ids < ROWS)

    is_class_token = (row_ids - (row_ids // TOKENS) * TOKENS) == 0
    class_mask = valid & is_class_token[:, None]
    weight = tl.load(
        weight_ptr + cols,
        mask=col_valid,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_valid,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    batch_ids = row_ids // TOKENS
    class_offsets = batch_ids[:, None] * HIDDEN + cols
    tl.store(
        class_bf16_ptr + class_offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=class_mask,
    )


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="0ff22f63", BLOCK_H=256, ROW_BLOCK=1, num_warps=1, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0 = inputs
    norm_shape = _as_shape(shape0)
    batch = int(arg1_1.shape[0])
    tokens = int(arg1_1.shape[1])
    hidden = int(arg1_1.shape[2])
    rows = int(arg0_1.shape[0])

    normalized = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    class_bf16 = torch.empty_strided(
        (batch, hidden),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        (batch, tokens, 1),
        (tokens, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _layernorm_selected_affine_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        normalized,
        class_bf16,
        div,
        ROWS=rows,
        TOKENS=tokens,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return normalized, class_bf16, div
