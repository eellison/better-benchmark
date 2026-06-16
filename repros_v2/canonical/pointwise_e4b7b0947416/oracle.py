"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full four-branch bf16 Inception BatchNorm-inference affine, Inductor-style fp32 ReLU, final bf16 output store, and fixed channel concatenation in one output-channel-tiled Triton kernel, whereas Inductor lowers the sibling broadcast pointwise producers and fixed aten.cat layout as a generic flattened scheduler fragment; Inductor cannot do this today because its scheduler does not expose a reusable cat-epilogue lowering for sibling pointwise producers with static channel intervals and per-branch BN parameters; the fix is SCHEDULER_FUSION: teach the scheduler to sink fixed channel-cat writes into branch pointwise codegen while preserving libdevice sqrt/reciprocal, eps, final-cast, ReLU, and channels-last output stride semantics."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _multi_bn_relu_cat_kernel(
    mean0_ptr,
    x0_ptr,
    var0_ptr,
    weight0_ptr,
    bias0_ptr,
    mean1_ptr,
    x1_ptr,
    var1_ptr,
    weight1_ptr,
    bias1_ptr,
    mean2_ptr,
    x2_ptr,
    var2_ptr,
    weight2_ptr,
    bias2_ptr,
    mean3_ptr,
    x3_ptr,
    var3_ptr,
    weight3_ptr,
    bias3_ptr,
    out_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    HW: tl.constexpr,
    OUT_C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    batch = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    valid_c = c < OUT_C
    valid_hw = hw_offsets < HW
    mask = valid_c[:, None] & valid_hw[None, :]

    c01: tl.constexpr = C0 + C1
    c012: tl.constexpr = C0 + C1 + C2
    in0 = c < C0
    in1 = (c >= C0) & (c < c01)
    in2 = (c >= c01) & (c < c012)
    in3 = c >= c012

    c0 = c
    c1 = c - C0
    c2 = c - c01
    c3 = c - c012

    x0_offsets = batch * C0 * HW + hw_offsets[None, :] * C0 + c0[:, None]
    x1_offsets = batch * C1 * HW + hw_offsets[None, :] * C1 + c1[:, None]
    x2_offsets = batch * C2 * HW + hw_offsets[None, :] * C2 + c2[:, None]
    x3_offsets = batch * C3 * HW + hw_offsets[None, :] * C3 + c3[:, None]

    x = tl.load(x0_ptr + x0_offsets, mask=mask & in0[:, None], other=0.0).to(tl.float32)
    x += tl.load(x1_ptr + x1_offsets, mask=mask & in1[:, None], other=0.0).to(tl.float32)
    x += tl.load(x2_ptr + x2_offsets, mask=mask & in2[:, None], other=0.0).to(tl.float32)
    x += tl.load(x3_ptr + x3_offsets, mask=mask & in3[:, None], other=0.0).to(tl.float32)

    mean = tl.load(mean0_ptr + c0, mask=valid_c & in0, other=0.0).to(tl.float32)
    mean += tl.load(mean1_ptr + c1, mask=valid_c & in1, other=0.0).to(tl.float32)
    mean += tl.load(mean2_ptr + c2, mask=valid_c & in2, other=0.0).to(tl.float32)
    mean += tl.load(mean3_ptr + c3, mask=valid_c & in3, other=0.0).to(tl.float32)

    var = tl.load(var0_ptr + c0, mask=valid_c & in0, other=0.0).to(tl.float32)
    var += tl.load(var1_ptr + c1, mask=valid_c & in1, other=0.0).to(tl.float32)
    var += tl.load(var2_ptr + c2, mask=valid_c & in2, other=0.0).to(tl.float32)
    var += tl.load(var3_ptr + c3, mask=valid_c & in3, other=0.0).to(tl.float32)

    weight = tl.load(weight0_ptr + c0, mask=valid_c & in0, other=0.0).to(tl.float32)
    weight += tl.load(weight1_ptr + c1, mask=valid_c & in1, other=0.0).to(tl.float32)
    weight += tl.load(weight2_ptr + c2, mask=valid_c & in2, other=0.0).to(tl.float32)
    weight += tl.load(weight3_ptr + c3, mask=valid_c & in3, other=0.0).to(tl.float32)

    bias = tl.load(bias0_ptr + c0, mask=valid_c & in0, other=0.0).to(tl.float32)
    bias += tl.load(bias1_ptr + c1, mask=valid_c & in1, other=0.0).to(tl.float32)
    bias += tl.load(bias2_ptr + c2, mask=valid_c & in2, other=0.0).to(tl.float32)
    bias += tl.load(bias3_ptr + c3, mask=valid_c & in3, other=0.0).to(tl.float32)

    invstd = 1.0 / libdevice.sqrt(var + 0.001)
    y = (x - mean[:, None]) * invstd[:, None]
    y = y * weight[:, None] + bias[:, None]
    relu = triton_helpers.maximum(tl.full([1], 0, tl.int32), y)

    out_offsets = batch * OUT_C * HW + hw_offsets[None, :] * OUT_C + c[:, None]
    tl.store(out_ptr + out_offsets, relu, mask=mask)


def _launch(inputs, *, BLOCK_C: int, BLOCK_HW: int):
    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        mean3,
        x3,
        var3,
        weight3,
        bias3,
    ) = inputs

    batch = int(x0.shape[0])
    height = int(x0.shape[2])
    width = int(x0.shape[3])
    hw = height * width
    channels0 = int(x0.shape[1])
    channels1 = int(x1.shape[1])
    channels2 = int(x2.shape[1])
    channels3 = int(x3.shape[1])
    out_channels = channels0 + channels1 + channels2 + channels3

    out = torch.empty_strided(
        (batch, out_channels, height, width),
        (out_channels * hw, 1, width * out_channels, out_channels),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    grid = (
        batch,
        triton.cdiv(out_channels, BLOCK_C),
        triton.cdiv(hw, BLOCK_HW),
    )
    _multi_bn_relu_cat_kernel[grid](
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        mean3,
        x3,
        var3,
        weight3,
        bias3,
        out,
        C0=channels0,
        C1=channels1,
        C2=channels2,
        C3=channels3,
        HW=hw,
        OUT_C=out_channels,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return out


# 25fb017b: four [128,192,17,17] branches -> [128,768,17,17]
# 78045192: [64,64,96,64] channel branches at 35x35 -> [128,288,35,35]
@oracle_impl(hardware="B200", point="25fb017b", BLOCK_C=32, BLOCK_HW=32)
@oracle_impl(hardware="B200", point="78045192", BLOCK_C=32, BLOCK_HW=32)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_HW: int):
    return _launch(inputs, BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW)
