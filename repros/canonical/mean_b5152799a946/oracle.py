"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Visformer inference residual-add, per-channel BatchNorm-affine, explicit bf16 affine cast, spatial mean, and final `[128, 768]` bf16 view in one channels-last Triton reduction kernel, whereas Inductor lowers the same normalization and mean scope through generic pointwise and reduction scheduling around a large rounded activation; Inductor cannot do this today because its norm lowering does not fuse the returned-only rounded affine producer into the immediate spatial reduction while preserving the bf16 cast boundaries and final view layout; the fix is SCHEDULER_FUSION: teach normalization scheduling to keep bf16 residual/affine round points live through small fixed spatial reductions and emit the final view directly."""

import torch
import triton
import triton.language as tl

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
def _f32_sqrt(a):
    return tl.inline_asm_elementwise(
        "sqrt.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_ROWS": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_ROWS": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_ROWS": 64}, num_warps=8, num_stages=3),
    ],
    key=["ROWS", "CHANNELS"],
)
@triton.jit
def _residual_bn_affine_spatial_mean_kernel(
    x0_ptr,
    x1_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    HW: tl.constexpr,
    STRIDE_N: tl.constexpr,
    STRIDE_C: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    spatial = tl.arange(0, BLOCK_HW)
    row_mask = rows < ROWS
    spatial_mask = spatial < HW

    n = rows // CHANNELS
    c = rows - n * CHANNELS
    h = spatial // WIDTH
    w = spatial - h * WIDTH
    offsets = (
        n[:, None] * STRIDE_N
        + c[:, None] * STRIDE_C
        + h[None, :] * STRIDE_H
        + w[None, :] * STRIDE_W
    )
    mask = row_mask[:, None] & spatial_mask[None, :]

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    added = _f32_add(x0, x1).to(tl.bfloat16)

    mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

    inv = _f32_div(1.0, _f32_sqrt(_f32_add(var, 1.0e-5)))
    normalized = _f32_mul(_f32_sub(added.to(tl.float32), mean[:, None]), inv[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[:, None]), bias[:, None])
    rounded = affine.to(tl.bfloat16)

    total = tl.sum(tl.where(spatial_mask[None, :], rounded.to(tl.float32), 0.0), axis=1)
    pooled = _f32_div(total, 49.0)
    tl.store(out_ptr + rows, pooled.to(tl.bfloat16), mask=row_mask)


@oracle_impl(hardware="B200", point="45e1ce96", BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_HW: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    n, channels, _height, width = arg0_1.shape
    out = torch.empty_strided(
        (n, channels),
        (channels, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = lambda meta: (triton.cdiv(n * channels, meta["BLOCK_ROWS"]),)
    _residual_bn_affine_spatial_mean_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out,
        ROWS=n * channels,
        CHANNELS=channels,
        WIDTH=width,
        HW=arg0_1.shape[2] * width,
        STRIDE_N=arg0_1.stride(0),
        STRIDE_C=arg0_1.stride(1),
        STRIDE_H=arg0_1.stride(2),
        STRIDE_W=arg0_1.stride(3),
        BLOCK_HW=BLOCK_HW,
    )
    return out
