"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 UNet inference-normalization, bf16-rounded ReLU, bilinear unsafe-index interpolation, right pad, and final `[1,1024,80,119]` cat scope in one output-tiled Triton kernel, whereas Inductor lowers the normalization producer, dtype-boundary activation, interpolation gathers, padding, and cat as separate generic pointwise/indexing regions; Inductor cannot do this today because its scheduler does not sink the channelwise affine/ReLU producer through fixed bilinear upsample and cat consumers while preserving the visible bf16 rounding and NaN behavior; the fix is SCHEDULER_FUSION: extend pointwise/indexing fusion to emit direct static upsample-cat stores from the normalized producer with exact dtype boundaries."""

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
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _bn_relu_value(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    offsets,
    channels,
    mask,
):
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)

    denom = _f32_sqrt(_f32_add(var, 1.0e-5))
    invstd = _f32_div(tl.full((offsets.shape[0],), 1.0, tl.float32), denom)
    norm = _f32_mul(_f32_sub(x, mean), invstd)
    affine = _f32_add(_f32_mul(norm, weight), bias)
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    return _relu_preserve_nan(rounded)


@triton.jit
def _upsample_cat_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    skip_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    UPSAMPLE_W: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < TOTAL

    ow = offsets % OUT_W
    t0 = offsets // OUT_W
    oh = t0 % OUT_H
    channel = (t0 // OUT_H) % (C + C)
    batch = t0 // (OUT_H * (C + C))
    first_half = channel < C
    branch_channel = channel - C

    skip_offsets = batch * C * OUT_H * OUT_W + channel * OUT_H * OUT_W + oh * OUT_W + ow
    skip_value = tl.load(skip_ptr + skip_offsets, mask=mask & first_half, other=0.0)

    valid_branch = mask & (~first_half) & (ow < UPSAMPLE_W)
    row_f = _f32_mul(oh.to(tl.float32), 0.4936708860759494)
    row_f = tl.maximum(row_f, 0.0)
    row0 = row_f.to(tl.int64)
    row1 = tl.minimum(row0 + 1, IN_H - 1)
    row_alpha = _f32_sub(row_f, row0.to(tl.float32))
    row_alpha = tl.maximum(tl.minimum(row_alpha, 1.0), 0.0)

    col_f = _f32_mul(ow.to(tl.float32), 0.49572649572649574)
    col_f = tl.maximum(col_f, 0.0)
    col0 = col_f.to(tl.int64)
    col1 = tl.minimum(col0 + 1, IN_W - 1)
    col_alpha = _f32_sub(col_f, col0.to(tl.float32))
    col_alpha = tl.maximum(tl.minimum(col_alpha, 1.0), 0.0)

    input_base = batch * C * IN_H * IN_W + branch_channel * IN_H * IN_W
    v00 = _bn_relu_value(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, input_base + row0 * IN_W + col0, branch_channel, valid_branch)
    v01 = _bn_relu_value(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, input_base + row0 * IN_W + col1, branch_channel, valid_branch)
    v10 = _bn_relu_value(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, input_base + row1 * IN_W + col0, branch_channel, valid_branch)
    v11 = _bn_relu_value(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, input_base + row1 * IN_W + col1, branch_channel, valid_branch)

    top = _f32_add(v00, _f32_mul(_f32_sub(v01, v00), col_alpha))
    bottom = _f32_add(v10, _f32_mul(_f32_sub(v11, v10), col_alpha))
    interp = _f32_add(top, _f32_mul(_f32_sub(bottom, top), row_alpha))
    branch_value = tl.where(ow < UPSAMPLE_W, interp.to(tl.bfloat16, fp_downcast_rounding="rtne"), 0.0)
    value = tl.where(first_half, skip_value, branch_value)
    tl.store(out_ptr + offsets, value, mask=mask)


# a2b68844: (T([512], bf16), T([1,512,40,59], bf16), T([512], bf16), T([512], bf16), T([512], bf16), T([1,512,80,119], bf16), S([80,1]))
@oracle_impl(hardware="B200", point="a2b68844", BLOCK_E=256, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_E: int, num_warps: int, num_stages: int):
    mean, x, var, weight, bias, skip, _shape_param_0 = inputs
    del _shape_param_0

    channels = 512
    out_h = 80
    out_w = 119
    out = torch.empty_strided(
        (1, channels + channels, out_h, out_w),
        ((channels + channels) * out_h * out_w, out_h * out_w, out_w, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    total = out.numel()
    _upsample_cat_kernel[(triton.cdiv(total, BLOCK_E),)](
        mean,
        x,
        var,
        weight,
        bias,
        skip,
        out,
        TOTAL=total,
        C=channels,
        IN_H=40,
        IN_W=59,
        OUT_H=out_h,
        OUT_W=out_w,
        UPSAMPLE_W=118,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
