"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete f32 affine LayerNorm scope in one shape-specialized Triton row-reduction kernel, including population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt, returned mean and rsqrt side outputs, broadcast scale/bias, final bf16 cast, and the metadata-only `[B*S, H]` view, whereas Inductor lowers the decomposed var_mean/rsqrt/affine/cast/view graph through its generic normalization scheduler; Inductor cannot do this today because the scheduler/codegen path does not select a guarded large-hidden affine-LayerNorm row tile that keeps the row resident while sinking live side-output stores and the bf16 viewed epilogue; the fix is SCHEDULER_FUSION: add a benchmark-gated large-hidden affine LayerNorm schedule that specializes hidden-size tiling, preserves required mean/invstd outputs, and writes the final bf16 viewed layout directly."""

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
def _layernorm_affine_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < HIDDEN)
    offsets = rows[:, None] * HIDDEN + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    col_mask = cols[None, :] < HIDDEN
    mean = tl.sum(tl.where(col_mask, x, 0.0), axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    variance = tl.sum(tl.where(col_mask, _f32_mul(centered, centered), 0.0), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))

    weight = tl.load(weight_ptr + cols, mask=cols < HIDDEN, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=cols < HIDDEN, other=0.0).to(tl.float32)
    normalized = _f32_mul(centered, invstd[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])

    row_mask = rows < ROWS
    tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(invstd_ptr + rows, invstd, mask=row_mask)
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


@triton.jit
def _layernorm_affine_h2560_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    EPSILON: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_mask = rows < ROWS
    cols0 = tl.arange(0, 2048)
    cols1 = tl.arange(0, 512) + 2048
    offsets0 = rows[:, None] * 2560 + cols0[None, :]
    offsets1 = rows[:, None] * 2560 + cols1[None, :]

    x0 = tl.load(x_ptr + offsets0, mask=row_mask[:, None], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + offsets1, mask=row_mask[:, None], other=0.0).to(tl.float32)
    mean = (tl.sum(x0, axis=1) + tl.sum(x1, axis=1)) / 2560.0

    centered0 = _f32_sub(x0, mean[:, None])
    centered1 = _f32_sub(x1, mean[:, None])
    variance = (
        tl.sum(_f32_mul(centered0, centered0), axis=1)
        + tl.sum(_f32_mul(centered1, centered1), axis=1)
    ) / 2560.0
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))

    weight0 = tl.load(weight_ptr + cols0).to(tl.float32)
    bias0 = tl.load(bias_ptr + cols0).to(tl.float32)
    weight1 = tl.load(weight_ptr + cols1).to(tl.float32)
    bias1 = tl.load(bias_ptr + cols1).to(tl.float32)

    norm0 = _f32_mul(centered0, invstd[:, None])
    norm1 = _f32_mul(centered1, invstd[:, None])
    affine0 = _f32_add(_f32_mul(norm0, weight0[None, :]), bias0[None, :])
    affine1 = _f32_add(_f32_mul(norm1, weight1[None, :]), bias1[None, :])

    tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(invstd_ptr + rows, invstd, mask=row_mask)
    tl.store(out_ptr + offsets0, affine0.to(tl.bfloat16), mask=row_mask[:, None])
    tl.store(out_ptr + offsets1, affine1.to(tl.bfloat16), mask=row_mask[:, None])


# (T([32,128,2560], f32), T([2560], f32), T([2560], f32), S([4096,2560]))
@oracle_impl(hardware="B200", point="23d9c217", BLOCK_H=4096, ROW_BLOCK=2, num_warps=8, num_stages=3)
# (T([16,128,2560], f32), T([2560], f32), T([2560], f32), S([2048,2560]))
@oracle_impl(hardware="B200", point="3d83c7c9", BLOCK_H=4096, ROW_BLOCK=2, num_warps=8, num_stages=3)
# (T([64,128,1024], f32), T([1024], f32), T([1024], f32), S([8192,1024]))
@oracle_impl(hardware="B200", point="08be8792", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
# (T([8,1024,1024], f32), T([1024], f32), T([1024], f32), S([8192,1024]))
@oracle_impl(hardware="B200", point="aeee3632", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
# (T([4,2048,768], f32), T([768], f32), T([768], f32), S([8192,768]))
@oracle_impl(hardware="B200", point="0f3a195f", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
# (T([128,128,1024], f32), T([1024], f32), T([1024], f32), S([16384,1024]))
@oracle_impl(hardware="B200", point="f990afad", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
# (T([32,128,1024], f32), T([1024], f32), T([1024], f32), S([4096,1024]))
@oracle_impl(hardware="B200", point="3220a3ed", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK, num_warps, num_stages):
    arg0_1, arg1_1, arg2_1, _shape_param_0 = inputs
    batch = int(arg0_1.shape[0])
    tokens = int(arg0_1.shape[1])
    hidden = int(arg0_1.shape[2])
    rows = batch * tokens
    out_shape = tuple(int(dim) for dim in _shape_param_0)

    mean = torch.empty((batch, tokens, 1), device=arg0_1.device, dtype=torch.float32)
    invstd = torch.empty((batch, tokens, 1), device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if hidden == 2560:
        _layernorm_affine_h2560_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            mean,
            invstd,
            out,
            ROWS=rows,
            EPSILON=EPS,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _layernorm_affine_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            mean,
            invstd,
            out,
            ROWS=rows,
            HIDDEN=hidden,
            EPSILON=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return mean, invstd, out
