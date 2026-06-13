"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 inference BatchNorm affine, explicit bf16 rounding, NaN-preserving ReLU, and 7x7 spatial mean directly into the final bf16 `[N,C]` tensor for both recorded MnasNet/ShuffleNet points while folding the per-channel affine to scale/shift inside each row reduction, whereas Inductor lowers the decomposed broadcast affine and small spatial reduction through a generic reduction schedule that preserves more per-element BN algebra; Inductor cannot do this today because its reduction codegen does not canonicalize inference-BN parameters into reusable affine scale/shift values for activation-fed spatial mean reductions while preserving the explicit bf16 cast boundary; the fix is ALGEBRAIC_ELIMINATION: canonicalize inference BN to per-channel scale/shift before lowering fixed-spatial activation reductions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


HW_SIZE = 49
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
def _affine_precompute_kernel(
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    affine_ptr,
    CHANNELS: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = channels < CHANNELS
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)

    var_eps = _f32_add(var, EPS_VALUE)
    sqrt = tl.sqrt_rn(var_eps)
    reciprocal = _f32_div(1.0, sqrt)
    invstd = _f32_mul(reciprocal, 1.0)
    scale = _f32_mul(invstd, weight)
    shift = _f32_sub(bias, _f32_mul(mean, scale))
    tl.store(affine_ptr + channels, scale, mask=mask)
    tl.store(affine_ptr + CHANNELS + channels, shift, mask=mask)


@triton.jit
def _bn_relu_spatial_mean_kernel(
    affine_ptr,
    x_ptr,
    out_ptr,
    TOTAL_ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    spatial = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    spatial_mask = spatial < HW
    mask = row_mask[:, None] & spatial_mask[None, :]

    channels = rows - (rows // CHANNELS) * CHANNELS
    x_offsets = rows[:, None] * HW + spatial[None, :]
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(affine_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    shift = tl.load(
        affine_ptr + CHANNELS + channels,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(x, scale[:, None]), shift[:, None])
    rounded = affine.to(tl.bfloat16)

    zero = tl.full((BLOCK_ROWS, BLOCK_HW), 0.0, tl.float32).to(tl.bfloat16)
    relu = tl.where(rounded < zero, zero, rounded)
    relu_f32 = tl.where(spatial_mask[None, :], relu.to(tl.float32), 0.0)
    reduced = _f32_div(tl.sum(relu_f32, axis=1), 49.0)
    tl.store(out_ptr + rows, reduced.to(tl.bfloat16), mask=row_mask)


def _launch(inputs, *, BLOCK_ROWS: int, num_warps: int):
    mean, x, var, weight, bias = inputs
    n_size = int(x.shape[0])
    channels = int(x.shape[1])
    total_rows = n_size * channels
    affine = torch.empty_strided(
        (2, channels),
        (channels, 1),
        device=x.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        (n_size, channels),
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _affine_precompute_kernel[(triton.cdiv(channels, 256),)](
        mean,
        var,
        weight,
        bias,
        affine,
        CHANNELS=channels,
        EPS_VALUE=EPS,
        BLOCK_C=256,
        num_warps=4,
        num_stages=3,
    )
    _bn_relu_spatial_mean_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        affine,
        x,
        output,
        TOTAL_ROWS=total_rows,
        CHANNELS=channels,
        HW=HW_SIZE,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=64,
        num_warps=num_warps,
        num_stages=3,
    )
    return output


# 8444bfb4: (T([1280], bf16), T([32,1280,7,7], bf16), T([1280], bf16), T([1280], bf16), T([1280], bf16))
@oracle_impl(hardware="B200", point="8444bfb4", BLOCK_ROWS=8, num_warps=1)
# 2adc7e85: (T([1024], bf16), T([64,1024,7,7], bf16), T([1024], bf16), T([1024], bf16), T([1024], bf16))
@oracle_impl(hardware="B200", point="2adc7e85", BLOCK_ROWS=8, num_warps=1)
def oracle_forward(inputs, *, BLOCK_ROWS: int, num_warps: int):
    return _launch(inputs, BLOCK_ROWS=BLOCK_ROWS, num_warps=num_warps)
