"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Inception six-branch BN-inference affine, explicit bf16 cast, NaN-preserving ReLU, fixed channel concatenation, returned channels-last cat tensor, and padded 3x3 stride-1 avg_pool2d tail, whereas Inductor lowers the sibling BN/ReLU branches, static channel cats, and pooling consumer as separate generic pointwise/cat/pool regions; Inductor cannot do this today because scheduler fusion does not sink fixed channel-cat BN/ReLU producers into pooling stencil codegen while also preserving the visible rounded cat output and channels-last output strides; the fix is SCHEDULER_FUSION: teach pointwise/layout/pooling scheduling to keep static Inception branch cats virtual across avg_pool2d and emit both the returned cat and pooled tensors with exact bf16 rounding boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
OUT_CHANNELS = 2048
HEIGHT = 8
WIDTH = 8
HW = 64
OUT_STRIDE = (OUT_CHANNELS * HW, 1, WIDTH * OUT_CHANNELS, OUT_CHANNELS)
EPS = 0.001


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
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return tl.where(x != x, x, rounded.to(tl.float32, bitcast=True))


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _branch_bn_relu_cat_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    cat_ptr,
    C: tl.constexpr,
    OUT_C_OFFSET: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    hw_offsets = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_offsets = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)

    hw_mask = hw_offsets < 64
    c_mask = c_offsets < C
    mask = hw_mask[:, None] & c_mask[None, :]

    x_offsets = batch * (C * 64) + hw_offsets[:, None] * C + c_offsets[None, :]
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[None, :])
    var_eps = _f32_add(var, 0.001)
    sqrt = tl.sqrt_rn(var_eps)
    invstd = _f32_mul(_f32_div(1.0, sqrt), 1.0)
    normed = _f32_mul(centered, invstd[None, :])
    scaled = _f32_mul(normed, weight[None, :])
    affine = _f32_add(scaled, bias[None, :])
    rounded = _round_bf16_to_fp32(affine)
    out = _relu_preserve_nan(rounded)

    out_offsets = (
        batch * (2048 * 64)
        + hw_offsets[:, None] * 2048
        + OUT_C_OFFSET
        + c_offsets[None, :]
    )
    tl.store(cat_ptr + out_offsets, out, mask=mask)


@triton.jit
def _avgpool3x3_channels_last_kernel(
    cat_ptr,
    avg_ptr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    hw_offsets = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_offsets = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)

    hw_mask = hw_offsets < 64
    c_mask = c_offsets < 2048
    oh = hw_offsets // 8
    ow = hw_offsets - oh * 8
    batch_base = batch * (2048 * 64)

    acc = tl.zeros((BLOCK_HW, BLOCK_C), tl.float32)
    for kh in tl.static_range(0, 3):
        ih = oh + kh - 1
        h_valid = (ih >= 0) & (ih < 8)
        for kw in tl.static_range(0, 3):
            iw = ow + kw - 1
            valid = hw_mask & h_valid & (iw >= 0) & (iw < 8)
            offsets = batch_base + (ih[:, None] * 8 + iw[:, None]) * 2048 + c_offsets[None, :]
            value = tl.load(
                cat_ptr + offsets,
                mask=valid[:, None] & c_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = _f32_add(acc, value)

    out_offsets = batch_base + hw_offsets[:, None] * 2048 + c_offsets[None, :]
    avg = _round_bf16_to_fp32(_f32_div(acc, 9.0))
    tl.store(avg_ptr + out_offsets, avg, mask=hw_mask[:, None] & c_mask[None, :])


def _launch_branch(
    mean,
    x,
    var,
    weight,
    bias,
    cat,
    *,
    channels,
    offset,
    block_hw,
    block_c,
    num_warps,
):
    grid = (BATCH, triton.cdiv(HW, block_hw), triton.cdiv(channels, block_c))
    _branch_bn_relu_cat_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        cat,
        C=channels,
        OUT_C_OFFSET=offset,
        BLOCK_HW=block_hw,
        BLOCK_C=block_c,
        num_warps=num_warps,
        num_stages=3,
    )


@oracle_impl(hardware="B200", point="13ab6fb6", BLOCK_HW=16, BLOCK_C=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int, num_warps: int):
    cat = torch.empty_strided(
        (BATCH, OUT_CHANNELS, HEIGHT, WIDTH),
        OUT_STRIDE,
        device=inputs[1].device,
        dtype=torch.bfloat16,
    )
    avg = torch.empty_strided(
        (BATCH, OUT_CHANNELS, HEIGHT, WIDTH),
        OUT_STRIDE,
        device=inputs[1].device,
        dtype=torch.bfloat16,
    )

    _launch_branch(*inputs[0:5], cat, channels=320, offset=0, block_hw=BLOCK_HW, block_c=BLOCK_C, num_warps=num_warps)
    _launch_branch(*inputs[5:10], cat, channels=384, offset=320, block_hw=BLOCK_HW, block_c=BLOCK_C, num_warps=num_warps)
    _launch_branch(*inputs[10:15], cat, channels=384, offset=704, block_hw=BLOCK_HW, block_c=BLOCK_C, num_warps=num_warps)
    _launch_branch(*inputs[15:20], cat, channels=384, offset=1088, block_hw=BLOCK_HW, block_c=BLOCK_C, num_warps=num_warps)
    _launch_branch(*inputs[20:25], cat, channels=384, offset=1472, block_hw=BLOCK_HW, block_c=BLOCK_C, num_warps=num_warps)
    _launch_branch(*inputs[25:30], cat, channels=192, offset=1856, block_hw=BLOCK_HW, block_c=BLOCK_C, num_warps=num_warps)

    grid = (BATCH, triton.cdiv(HW, BLOCK_HW), triton.cdiv(OUT_CHANNELS, BLOCK_C))
    _avgpool3x3_channels_last_kernel[grid](
        cat,
        avg,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return cat, avg
