"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 BatchNorm-affine/ReLU reduction fragment by tiling the channels-last producer once, preserving the explicit bf16 affine, product, spatial-sum, gated-output, and final channel-sum rounding boundaries, writing the returned f32 gate tensor, returned bf16 ReLU activation, returned bf16 gated spatial sum, and final f32 channel vector; Inductor cannot do this today because it schedules the broadcast affine/ReLU producer, spatial reduction, gate epilogue, returned tensor stores, and downstream channel reduction as separate generic kernels over materialized intermediates instead of one full-scope producer-plus-reduction plan; the fix is SCHEDULER_FUSION: add a MobileNetV3 template that keeps the channels-last pointwise producer, dependent small spatial reduction, and sibling batch-channel finalization in one planned lowering while respecting bf16 cast boundaries."""

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
def _producer_spatial_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    arg7_ptr,
    out0_ptr,
    relu_out_ptr,
    gate_out_ptr,
    arg1_s0: tl.constexpr,
    arg1_s1: tl.constexpr,
    arg1_s2: tl.constexpr,
    arg1_s3: tl.constexpr,
    arg6_s0: tl.constexpr,
    arg6_s1: tl.constexpr,
    arg6_s2: tl.constexpr,
    arg6_s3: tl.constexpr,
    relu_s0: tl.constexpr,
    relu_s1: tl.constexpr,
    relu_s2: tl.constexpr,
    relu_s3: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)

    hw = tl.arange(0, BLOCK_HW)[:, None]
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    c_vec = c_block * BLOCK_C + tl.arange(0, BLOCK_C)

    hw_mask = hw < H * W
    c_mask = c < C
    elem_mask = hw_mask & c_mask
    c_vec_mask = c_vec < C

    h = hw // W
    w = hw - h * W
    x_offsets = n * arg1_s0 + c * arg1_s1 + h * arg1_s2 + w * arg1_s3
    grad_offsets = n * arg6_s0 + c * arg6_s1 + h * arg6_s2 + w * arg6_s3
    relu_offsets = n * relu_s0 + c * relu_s1 + h * relu_s2 + w * relu_s3

    x = tl.load(arg1_ptr + x_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    grad = tl.load(arg6_ptr + grad_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    mean = tl.load(arg2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(arg3_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(arg4_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(arg5_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    affine = _f32_add(scaled, bias)
    affine_bf16 = _round_bf16_to_f32(affine)
    relu = tl.where(affine_bf16 != affine_bf16, affine_bf16, tl.maximum(affine_bf16, 0.0))
    tl.store(relu_out_ptr + relu_offsets, relu, mask=elem_mask)

    product_bf16 = _round_bf16_to_f32(_f32_mul(grad, relu))
    spatial_sum = tl.sum(tl.where(elem_mask, product_bf16, 0.0), axis=0)
    spatial_sum_bf16 = _round_bf16_to_f32(spatial_sum)

    gate = tl.load(arg0_ptr + n * C + c_vec, mask=c_vec_mask, other=0.0).to(tl.float32)
    tl.store(out0_ptr + n * C + c_vec, gate, mask=c_vec_mask)

    scaled_sum = _f32_mul(spatial_sum_bf16, 0.16666666666666666)
    fill = tl.load(arg7_ptr).to(tl.float32)
    gated = tl.where((gate > -3.0) & (gate < 3.0), scaled_sum, fill)
    tl.store(gate_out_ptr + n * C + c_vec, gated, mask=c_vec_mask)


@triton.jit
def _final_channel_sum_kernel(
    gate_out_ptr,
    out_ptr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    n = tl.arange(0, BLOCK_N)[:, None]
    active = (c < C) & (n < N)
    vals = tl.load(gate_out_ptr + n * C + c, mask=active, other=0.0).to(tl.float32)
    reduced = tl.sum(vals, axis=0)
    reduced_bf16 = _round_bf16_to_f32(reduced)

    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(out_ptr + c_vec, reduced_bf16, mask=c_vec < C)


def _channels_last_stride(n: int, c: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (c * h * w, 1, c * w, c)


@oracle_impl(hardware="B200", point="c557d2c1", BLOCK_HW=1024, BLOCK_C=16, FINAL_BLOCK_C=64, num_warps=8)
@oracle_impl(hardware="B200", point="0e83eb42", BLOCK_HW=1024, BLOCK_C=16, FINAL_BLOCK_C=64, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 = inputs
    n = int(arg1.shape[0])
    c = int(arg1.shape[1])
    h = int(arg1.shape[2])
    w = int(arg1.shape[3])

    out0 = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=arg1.device,
        dtype=torch.float32,
    )
    relu_out = torch.empty_strided(
        (n, c, h, w),
        _channels_last_stride(n, c, h, w),
        device=arg1.device,
        dtype=torch.bfloat16,
    )
    gate_out = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=arg1.device,
        dtype=torch.bfloat16,
    )
    channel_out = torch.empty((c,), device=arg1.device, dtype=torch.float32)

    _producer_spatial_kernel[(n, triton.cdiv(c, BLOCK_C))](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        out0,
        relu_out,
        gate_out,
        arg1_s0=arg1.stride(0),
        arg1_s1=arg1.stride(1),
        arg1_s2=arg1.stride(2),
        arg1_s3=arg1.stride(3),
        arg6_s0=arg6.stride(0),
        arg6_s1=arg6.stride(1),
        arg6_s2=arg6.stride(2),
        arg6_s3=arg6.stride(3),
        relu_s0=relu_out.stride(0),
        relu_s1=relu_out.stride(1),
        relu_s2=relu_out.stride(2),
        relu_s3=relu_out.stride(3),
        C=c,
        H=h,
        W=w,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _final_channel_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        gate_out,
        channel_out,
        C=c,
        N=n,
        BLOCK_N=32,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    return out0, relu_out, gate_out, channel_out
