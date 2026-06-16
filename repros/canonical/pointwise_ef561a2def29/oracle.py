"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 BN-inference affine, explicit bf16 cast, NaN-preserving ReLU, and padded 3x3 stride-2 maxpool value stencil in one Triton kernel; whereas Inductor materializes the full bf16 ReLU activation before a separate low-memory maxpool-with-offsets region; Inductor cannot do this today because scheduler fusion does not sink broadcast pointwise producers through padded pooling stencils while dropping the unused offsets output; the fix is SCHEDULER_FUSION: inline affine/ReLU producers into low-memory maxpool value codegen with exact bf16 cast and NaN semantics."""

import torch
import triton
import triton.language as tl

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
def _f32_rcp(a):
    return tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _bn_relu_maxpool_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BATCH: tl.constexpr,
    CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    OUT_HEIGHT: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_O: tl.constexpr,
    EPS_: tl.constexpr,
):
    n = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    out_linear = tl.program_id(2) * BLOCK_O + tl.arange(0, BLOCK_O)
    oh = out_linear // OUT_WIDTH
    ow = out_linear - oh * OUT_WIDTH

    channel_mask = channels < CHANNELS
    out_mask = out_linear < (OUT_HEIGHT * OUT_WIDTH)

    mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=channel_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

    invstd = _f32_rcp(_f32_sqrt(_f32_add(var, EPS_)))
    input_base = n * CHANNELS * HEIGHT * WIDTH + channels[:, None] * HEIGHT * WIDTH
    best = tl.full((BLOCK_C, BLOCK_O), -float("inf"), tl.float32)

    for kh in tl.static_range(0, 3):
        ih = oh * 2 + kh - 1
        valid_h = (ih >= 0) & (ih < HEIGHT)
        for kw in tl.static_range(0, 3):
            iw = ow * 2 + kw - 1
            valid_w = (iw >= 0) & (iw < WIDTH)
            valid = (
                channel_mask[:, None]
                & out_mask[None, :]
                & valid_h[None, :]
                & valid_w[None, :]
            )
            x = tl.load(
                x_ptr + input_base + ih[None, :] * WIDTH + iw[None, :],
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            y = _f32_mul(_f32_sub(x, mean[:, None]), invstd[:, None])
            y = _f32_mul(y, weight[:, None])
            y = _f32_add(y, bias[:, None])
            y = _relu_preserve_nan(_round_bf16_to_f32(y))
            take = valid & ((y > best) | (y != y))
            best = tl.where(take, y, best)

    out_offsets = (
        n * CHANNELS * OUT_HEIGHT * OUT_WIDTH
        + channels[:, None] * OUT_HEIGHT * OUT_WIDTH
        + out_linear[None, :]
    )
    tl.store(out_ptr + out_offsets, best, mask=channel_mask[:, None] & out_mask[None, :])


@oracle_impl(hardware="B200", point="e4de5f8d", BLOCK_C=4, BLOCK_O=64, num_warps=4)
@oracle_impl(hardware="B200", point="4194c732", BLOCK_C=4, BLOCK_O=64, num_warps=4)
@oracle_impl(hardware="B200", point="9bacc993", BLOCK_C=4, BLOCK_O=64, num_warps=4)
@oracle_impl(hardware="B200", point="aff40914", BLOCK_C=4, BLOCK_O=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_O: int, num_warps: int):
    mean, x, var, weight, bias = inputs[:5]
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    out_height = (height + 2 - 3) // 2 + 1
    out_width = (width + 2 - 3) // 2 + 1
    out = torch.empty_strided(
        (batch, channels, out_height, out_width),
        (channels * out_height * out_width, out_height * out_width, out_width, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _bn_relu_maxpool_kernel[
        (
            batch,
            triton.cdiv(channels, BLOCK_C),
            triton.cdiv(out_height * out_width, BLOCK_O),
        )
    ](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        BATCH=batch,
        CHANNELS=channels,
        HEIGHT=height,
        WIDTH=width,
        OUT_HEIGHT=out_height,
        OUT_WIDTH=out_width,
        BLOCK_C=BLOCK_C,
        BLOCK_O=BLOCK_O,
        EPS_=EPS,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
