"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete LearningToPaint inference BatchNorm-affine, explicit bf16 cast, bf16 residual add, NaN-preserving ReLU, fixed 4x4 avg_pool2d, and final `[96, 512]` bf16 view in one Triton reduction kernel, whereas Inductor lowers the affine producer, residual/ReLU pointwise chain, and downstream pool/view as generic scheduled regions around a full rounded activation; Inductor cannot do this today because scheduler fusion does not sink a normalization lowering producer with required bf16 cast boundaries into a fixed-window pooling consumer that only returns the pooled view; the fix is SCHEDULER_FUSION: teach norm/pooling scheduling to keep rounded BN-residual-ReLU values virtual through small spatial average pools and emit the final view directly."""

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


@triton.jit
def _relu_preserve_nan_bf16(x):
    zero = tl.full(x.shape, 0.0, tl.bfloat16)
    return tl.where((x > zero) | (x != x), x, zero)


@triton.jit
def _bn_residual_relu_avgpool_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    spatial = tl.arange(0, BLOCK_HW)
    row_mask = rows < ROWS
    spatial_mask = spatial < HW
    mask = row_mask[:, None] & spatial_mask[None, :]

    channel = rows - (rows // CHANNELS) * CHANNELS
    offsets = rows[:, None] * HW + spatial[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    mean = tl.load(mean_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)

    inv = _f32_div(1.0, _f32_sqrt(_f32_add(var, 1.0e-5)))
    normalized = _f32_mul(_f32_sub(x, mean[:, None]), inv[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[:, None]), bias[:, None])
    affine_bf16 = affine.to(tl.bfloat16)
    added = (affine_bf16 + residual).to(tl.bfloat16)
    relu = _relu_preserve_nan_bf16(added)

    total = tl.sum(tl.where(spatial_mask[None, :], relu.to(tl.float32), 0.0), axis=1)
    pooled = _f32_div(total, 16.0)
    tl.store(out_ptr + rows, pooled.to(tl.bfloat16), mask=row_mask)


@oracle_impl(hardware="B200", point="526a63fd", BLOCK_ROWS=16, BLOCK_HW=16, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_ROWS: int,
    BLOCK_HW: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0 = inputs
    del _shape_param_0
    batch, channels, height, width = arg1_1.shape
    out = torch.empty_strided(
        (batch, channels),
        (channels, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    rows = batch * channels
    _bn_residual_relu_avgpool_kernel[(triton.cdiv(rows, BLOCK_ROWS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out,
        ROWS=rows,
        CHANNELS=channels,
        HW=height * width,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
