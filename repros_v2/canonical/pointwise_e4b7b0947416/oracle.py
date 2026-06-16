"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full four-branch bf16 Inception BatchNorm-inference affine, explicit bf16 cast, NaN-preserving ReLU, and channel concatenation by writing each branch directly into the final channels-last bf16 output, whereas Inductor lowers the sibling broadcast pointwise producers and fixed aten.cat layout as generic scheduler fragments; Inductor cannot do this today because its scheduler does not model aten.cat as an output-layout epilogue for sibling pointwise producers with static channel intervals and preserved bf16 cast boundaries; the fix is SCHEDULER_FUSION: teach the scheduler to sink fixed channel-cat writes into the branch pointwise codegen while preserving the sqrt/reciprocal, eps, bf16 cast, ReLU, and output stride semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_bf16_preserve_nan(x):
    zero = tl.full(x.shape, 0.0, tl.float32).to(tl.bfloat16)
    return tl.where((x > zero) | (x != x), x, zero)


@triton.jit
def _branch_bn_relu_cat_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    channels: tl.constexpr,
    hw: tl.constexpr,
    out_channels: tl.constexpr,
    out_c_offset: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    batch = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = (c_offsets[:, None] < channels) & (hw_offsets[None, :] < hw)

    x_offsets = batch * channels * hw + hw_offsets[None, :] * channels + c_offsets[:, None]
    out_offsets = (
        batch * out_channels * hw
        + hw_offsets[None, :] * out_channels
        + out_c_offset
        + c_offsets[:, None]
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)

    invstd = 1.0 / tl.sqrt(var[:, None] + 0.001)
    y = (x - mean[:, None]) * invstd
    y = y * weight[:, None] + bias[:, None]
    y_bf16 = y.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + out_offsets, _relu_bf16_preserve_nan(y_bf16), mask=mask)


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

    def run_branch(mean, x, var, weight, bias, channels, out_c_offset):
        grid = (
            batch,
            triton.cdiv(channels, BLOCK_C),
            triton.cdiv(hw, BLOCK_HW),
        )
        _branch_bn_relu_cat_kernel[grid](
            mean,
            x,
            var,
            weight,
            bias,
            out,
            channels=channels,
            hw=hw,
            out_channels=out_channels,
            out_c_offset=out_c_offset,
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=3,
        )

    run_branch(mean0, x0, var0, weight0, bias0, channels0, 0)
    run_branch(mean1, x1, var1, weight1, bias1, channels1, channels0)
    run_branch(mean2, x2, var2, weight2, bias2, channels2, channels0 + channels1)
    run_branch(
        mean3,
        x3,
        var3,
        weight3,
        bias3,
        channels3,
        channels0 + channels1 + channels2,
    )
    return out


# 25fb017b: four [128,192,17,17] branches -> [128,768,17,17]
# 78045192: [64,64,96,64] channel branches at 35x35 -> [128,288,35,35]
@oracle_impl(hardware="B200", point="25fb017b", BLOCK_C=32, BLOCK_HW=32)
@oracle_impl(hardware="B200", point="78045192", BLOCK_C=32, BLOCK_HW=32)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_HW: int):
    return _launch(inputs, BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW)
