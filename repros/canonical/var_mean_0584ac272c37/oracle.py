"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Swin singleton-window-reverse residual bf16 LayerNorm scope in one Triton row kernel, including the rank-changing [128,49,1024] -> [128,7,7,1024] -> [128,1,1,7,7,1024] -> permute -> [128,7,7,1024] chain, the bf16 residual add returned as [128,49,1024], the explicit bf16-to-fp32 cast before population var_mean(..., dim=2, correction=0, keepdim=True), eps=1e-5 rsqrt affine epilogue with bf16 weight/bias promotion, final bf16 cast, and [6272,1024] view output, whereas Inductor carries the singleton Swin window-reverse metadata through generic normalization scheduling and materializes the returned pre-norm tensor separately; Inductor cannot do this today because its view/permute simplifier does not prove the swapped singleton grid dimensions equivalent to an identity row layout before the normalization template runs while also preserving the returned bf16 side output; the fix is ALGEBRAIC_ELIMINATION: canonicalize singleton-grid Swin window-reverse reshape/permute chains to the identity row map, then emit the returned pre-norm store and normalized bf16 output from one LayerNorm row schedule."""

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
def _swin_singleton_window_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (row_ids < ROWS) & (cols < HIDDEN)
    offsets = row_ids * HIDDEN + cols

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

    add_bf16 = _f32_add(residual, flat).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)

    x = add_bf16.to(tl.float32)
    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=1)[:, None] / HIDDEN
    centered = _f32_sub(x, mean)
    centered_masked = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1)[:, None] / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    tl.store(
        norm_out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="d965afe2", BLOCK_H=1024, ROW_BLOCK=4, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2, _shape3, _shape4, _shape5 = inputs
    batch = int(arg1_1.shape[0])
    height = int(arg1_1.shape[1])
    width = int(arg1_1.shape[2])
    hidden = int(arg1_1.shape[3])
    tokens = height * width
    rows = batch * tokens
    add_shape = (batch, tokens, hidden)
    norm_shape = (rows, hidden)

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _swin_singleton_window_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out
