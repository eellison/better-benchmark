"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete OPT bf16 residual-add LayerNorm scope in one fixed-hidden Triton row kernel, including the returned `[4,2048,768]` bf16 residual view, the explicit bf16 residual-add rounding before fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt affine epilogue, final bf16 cast, and three `[8192,768]` alias views of one normalized output buffer, whereas Inductor lowers the captured graph through its generic persistent row-normalization path and repeated view metadata; Inductor cannot do this today because the normalization scheduler does not expose a full-scope residual-add LayerNorm plan that preserves the visible rounded residual producer together with the alias-only output contract; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to fuse visible residual producers, preserve explicit bf16 cast boundaries, and emit repeated view aliases from one planned row-normalization store."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
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
def _residual_layernorm_alias_kernel(
    arg0_ptr,
    arg1_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_ids[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (row_ids[:, None] < ROWS_) & (cols < HIDDEN_)
    col_mask = cols < HIDDEN_
    offsets = rows * HIDDEN_ + cols

    lhs = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_f32 = _f32_add(lhs, rhs)
    add_bf16 = add_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
    x = add_bf16.to(tl.float32)
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)

    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=1)[:, None] / HIDDEN_
    centered = _f32_sub(x, mean)
    centered_masked = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1)[:, None] / HIDDEN_
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_))
    normalized = _f32_mul(centered, invstd)

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    out = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(norm_out_ptr + offsets, out, mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# e4faf4aa: (T([8192,768], bf16), T([8192,768], bf16), T([768], bf16), T([768], bf16), S([4,2048,768]), ...)
@oracle_impl(hardware="B200", point="e4faf4aa", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    add_shape = _shape_tuple(shape0)
    norm_shape = _shape_tuple(shape1)

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        (int(arg0_1.shape[1]), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _residual_layernorm_alias_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_out,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        EPS_=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out, norm_out.view(_shape_tuple(shape2)), norm_out.view(_shape_tuple(shape3))
