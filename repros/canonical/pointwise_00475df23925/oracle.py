"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 inference BatchNorm-affine, explicit bf16 ReLU, cat, channel-shuffle clone layout, final view, and both returned split views by writing one shared contiguous bf16[64,464,7,7] backing tensor directly, whereas Inductor lowers the BN/ReLU producer and the cat/view/permute/clone/view/split layout transform as generic scheduled work with avoidable intermediate layout traffic; Inductor cannot do this today because its scheduler does not keep a fixed channel cat virtual across the channel-shuffle clone and split-return boundary while sinking the pointwise BN epilogue into the same output layout; the fix is SCHEDULER_FUSION: teach scheduler/codegen to represent fixed channel-shuffle cats as direct multi-source output layouts and fuse affine pointwise producers into the final split backing allocation."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 64
C = 232
H = 7
W = 7
HW = H * W
OUT_C = 2 * C


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
def _invstd_kernel(
    var_ptr,
    invstd_ptr,
    channels: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c_offsets < channels

    var = tl.load(var_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    var_eps = _f32_add(var, 1.0e-5)
    sqrt = tl.sqrt_rn(var_eps)
    recip = _f32_div(1.0, sqrt)
    invstd = _f32_mul(recip, 1.0)
    tl.store(invstd_ptr + c_offsets, invstd, mask=mask)


@triton.jit
def _shuffle_bn_relu_kernel(
    mean_ptr,
    x_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    skip_ptr,
    out_ptr,
    total: tl.constexpr,
    channels: tl.constexpr,
    out_channels: tl.constexpr,
    hw_size: tl.constexpr,
    x_stride_n: tl.constexpr,
    skip_stride_n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    hw_idx = offsets % hw_size
    src_channel = (offsets // hw_size) % channels
    batch = offsets // (channels * hw_size)

    skip_offsets = batch * skip_stride_n + src_channel * hw_size + hw_idx
    x_offsets = batch * x_stride_n + src_channel * hw_size + hw_idx

    skip = tl.load(skip_ptr + skip_offsets, mask=mask, other=0.0)

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + src_channel, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + src_channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + src_channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + src_channel, mask=mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    rounded = affine.to(tl.bfloat16)
    relu = tl.where(rounded.to(tl.float32) < 0.0, 0.0, rounded.to(tl.float32)).to(tl.bfloat16)

    out_base = batch * out_channels * hw_size + src_channel * (2 * hw_size) + hw_idx
    tl.store(out_ptr + out_base, skip, mask=mask)
    tl.store(out_ptr + out_base + hw_size, relu, mask=mask)


# 46084af7: (T([232], bf16), T([64,232,7,7], bf16), ..., strided skip branch)
@oracle_impl(hardware="B200", point="46084af7", BLOCK_C=256, BLOCK_OUT=512)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_OUT: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1 = inputs

    invstd = torch.empty((C,), device=arg1_1.device, dtype=torch.float32)
    shuffled = torch.empty_strided(
        (N, OUT_C, H, W),
        (OUT_C * HW, HW, W, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _invstd_kernel[(triton.cdiv(C, BLOCK_C),)](
        arg2_1,
        invstd,
        channels=C,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _shuffle_bn_relu_kernel[(triton.cdiv(N * C * HW, BLOCK_OUT),)](
        arg0_1,
        arg1_1,
        invstd,
        arg3_1,
        arg4_1,
        arg5_1,
        shuffled,
        total=N * C * HW,
        channels=C,
        out_channels=OUT_C,
        hw_size=HW,
        x_stride_n=C * HW,
        skip_stride_n=OUT_C * HW,
        BLOCK=BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )

    return shuffled[:, :C, :, :], shuffled[:, C:, :, :]
