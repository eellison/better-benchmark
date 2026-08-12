"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV2 bf16 BN-inference affine plus ReLU6 scope, including fp32 broadcast math, the intermediate bf16 rounding before clamp, NaN-preserving clamp_min/clamp_max, final bf16 output, and the captured contiguous or channels-last strides in one Triton storage-order pointwise kernel; whereas Inductor lowers the decomposed unsqueeze/sub/sqrt/reciprocal/mul/add/cast/clamp graph through a generic pointwise schedule that repeatedly handles broadcasted channel vectors; Inductor cannot do this today because its pointwise scheduler does not select a layout-aware per-channel BN-affine plus activation template for this full output contract; the fix is SCHEDULER_FUSION: add a guarded MobileNet BN-affine ReLU6 pointwise schedule that streams dense storage order while reusing the channel broadcast structure and preserving bf16 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bn_relu6_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < TOTAL
    if CHANNELS_LAST:
        c = offsets % C
    else:
        c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    denom = tl.sqrt(_f32_add(var, 1.0e-5))
    invstd = 1.0 / denom
    normed = _f32_mul(centered, _f32_mul(invstd, 1.0))
    scaled = _f32_mul(normed, weight)
    affine = _f32_add(scaled, bias)
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    relu = tl.where(rounded != rounded, rounded, tl.maximum(rounded, 0.0))
    relu6 = tl.where(relu != relu, relu, tl.minimum(relu, 6.0))
    tl.store(out_ptr + offsets, relu6.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@oracle_impl(hardware="B200", point="8a10670c", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="82cef4c9", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="2cf7960f", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="54640677", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="1ee4ce98", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="ba8126d9", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="ebe204a7", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="51719261", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="d53a7e50", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="2e1844e5", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="9c97edfa", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="3bdac97c", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="66878456", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="0598891b", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="91334d30", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="e5f12ec5", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="94494857", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="1cc752a4", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="92d91bac", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="3de61eda", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="8dd8c35d", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="522f10e2", BLOCK_N=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N, num_warps):
    mean, x, var, weight, bias = inputs
    _, channels, height, width = x.shape
    hw = height * width
    channels_last = x.stride(1) == 1
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    _bn_relu6_kernel[(triton.cdiv(x.numel(), BLOCK_N),)](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        TOTAL=x.numel(),
        C=channels,
        HW=hw,
        CHANNELS_LAST=channels_last,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out
