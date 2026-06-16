"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNetV2 inference BN-affine, explicit bf16 ReLU, virtual cat, fixed channel-shuffle clone layout, final view, and both returned split views by writing one shared contiguous bf16 backing tensor directly, whereas Inductor lowers the BN/ReLU producer and the cat/view/permute/clone/view/split layout indexing as generic scheduled work with avoidable intermediate layout traffic; Inductor cannot do this today because its scheduler does not keep a channel cat virtual across a fixed channel-shuffle clone and split-return boundary while fusing the pointwise BN epilogue into the same output layout; the fix is SCHEDULER_FUSION: teach scheduler/codegen to represent fixed channel-shuffle cats as a direct multi-source output layout and sink affine pointwise producers into the final split backing allocation."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _shuffle_bn_relu_kernel(
    mean_ptr,
    conv_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    left_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    OUT_C: tl.constexpr,
    LEFT_STRIDE_N: tl.constexpr,
    LEFT_STRIDE_C: tl.constexpr,
    LEFT_STRIDE_H: tl.constexpr,
    LEFT_STRIDE_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    safe_offsets = tl.where(mask, offsets, 0)

    hw = safe_offsets % HW
    out_c = (safe_offsets // HW) % OUT_C
    n = safe_offsets // (OUT_C * HW)
    c = out_c // 2
    use_bn = (out_c - c * 2) != 0

    h = hw // W
    w = hw - h * W
    conv_offsets = n * (C * HW) + c * HW + hw
    left_offsets = n * LEFT_STRIDE_N + c * LEFT_STRIDE_C + h * LEFT_STRIDE_H + w * LEFT_STRIDE_W

    left = tl.load(left_ptr + left_offsets, mask=mask & ~use_bn, other=0.0)
    x = tl.load(conv_ptr + conv_offsets, mask=mask & use_bn, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask & use_bn, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask & use_bn, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask & use_bn, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask & use_bn, other=0.0).to(tl.float32)

    centered = x - mean
    invstd = tl.full((BLOCK,), 1, tl.int32) / libdevice.sqrt(var + 1.0e-5)
    y = ((centered * invstd) * 1.0) * weight + bias
    y = _relu_preserve_nan(y.to(tl.bfloat16).to(tl.float32))
    out = tl.where(use_bn, y, left)
    tl.store(out_ptr + offsets, out, mask=mask)


def _launch(inputs, *, BLOCK: int):
    mean, conv, var, weight, bias, left, _shape0, _shape1 = inputs
    n = int(conv.shape[0])
    c = int(conv.shape[1])
    h = int(conv.shape[2])
    w = int(conv.shape[3])
    hw = h * w
    out_c = 2 * c
    out = torch.empty((n, out_c, h, w), device=conv.device, dtype=torch.bfloat16)
    total = n * out_c * hw
    _shuffle_bn_relu_kernel[(triton.cdiv(total, BLOCK),)](
        mean,
        conv,
        var,
        weight,
        bias,
        left,
        out,
        TOTAL=total,
        C=c,
        H=h,
        W=w,
        HW=hw,
        OUT_C=out_c,
        LEFT_STRIDE_N=left.stride(0),
        LEFT_STRIDE_C=left.stride(1),
        LEFT_STRIDE_H=left.stride(2),
        LEFT_STRIDE_W=left.stride(3),
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return out[:, :c, :, :], out[:, c:, :, :]


@oracle_impl(hardware="B200", point="1e6a9948", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    return _launch(inputs, BLOCK=BLOCK)
