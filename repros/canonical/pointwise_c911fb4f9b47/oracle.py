"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 channel-cat plus BN-inference affine ReLU scope, including the returned contiguous cat tensor and the returned bf16 ReLU tensor, from the two original NCHW sources in one Triton pointwise kernel. Inductor can fuse this specific captured graph, but it still represents the scheduler-fusion shape this row is measuring: fixed channel concat must be kept as a virtual producer while the per-channel fp32 normalization side inputs feed the downstream pointwise epilogue and both observable outputs are written. The fix is SCHEDULER_FUSION: keep fixed channel concat producers inline across broadcast pointwise consumers while preserving the output count, strides, NaN behavior, and final bf16 store boundary."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

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
def _cat_bn_relu_kernel(
    x0_ptr,
    x1_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    cat_ptr,
    relu_ptr,
    TOTAL: tl.constexpr,
    C1: tl.constexpr,
    C_TOTAL: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    spatial = offsets % HW
    channel = (offsets // HW) % C_TOTAL
    batch = offsets // (C_TOTAL * HW)

    from_x0 = channel < 16
    c1 = channel - 16
    x0_offsets = batch * 16 * HW + channel * HW + spatial
    x1_offsets = batch * C1 * HW + c1 * HW + spatial

    x0 = tl.load(x0_ptr + x0_offsets, mask=mask & from_x0, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask & ~from_x0, other=0.0).to(tl.float32)
    x = tl.where(from_x0, x0, x1)

    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    denom = libdevice.sqrt(_f32_add(var, 1.0e-5))
    invstd = _f32_mul(_f32_div(1.0, denom), 1.0)
    normed = _f32_mul(centered, invstd)
    scaled = _f32_mul(normed, weight)
    affine = _f32_add(scaled, bias)
    relu = tl.where(affine != affine, affine, tl.maximum(affine, 0.0))

    tl.store(cat_ptr + offsets, x, mask=mask)
    tl.store(relu_ptr + offsets, relu, mask=mask)


@oracle_impl(hardware="B200", point="286b5304", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="c7103f45", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="d46fd5bf", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="bdfe0800", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="3ba43305", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="ca621cc5", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="02da0370", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="5ec16488", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="1f781a5e", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="fe2f0933", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="b1d4084b", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="e08032de", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="1b282c91", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="7a77a894", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="5911c005", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="519fe36e", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="d64fa266", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="3dde721a", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="d99ca30a", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="9b5495c2", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x0, x1, mean, var, weight, bias = inputs
    batch = x0.shape[0]
    height = x0.shape[2]
    width = x0.shape[3]
    hw = height * width
    c1 = x1.shape[1]
    channels = 16 + c1
    shape = (batch, channels, height, width)
    stride = (channels * hw, hw, width, 1)

    cat = torch.empty_strided(shape, stride, device=x0.device, dtype=torch.bfloat16)
    relu = torch.empty_strided(shape, stride, device=x0.device, dtype=torch.bfloat16)
    total = batch * channels * hw
    _cat_bn_relu_kernel[(triton.cdiv(total, BLOCK),)](
        x0,
        x1,
        mean,
        var,
        weight,
        bias,
        cat,
        relu,
        TOTAL=total,
        C1=c1,
        C_TOTAL=channels,
        HW=hw,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return cat, relu
