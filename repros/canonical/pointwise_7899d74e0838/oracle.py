"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full bf16 BN-inference affine plus ReLU pointwise region with a layout-aware channel/spatial Triton tile that reuses each channel's mean, variance, weight, and bias values across the tile and preserves the dense NCHW or channels-last output strides, whereas Inductor emits a generic fused pointwise kernel for the decomposed unsqueeze/sub/sqrt/reciprocal/mul/add/cast/ReLU graph and repeats channel-invariant scalar work inside the element loop; Inductor cannot do this today because its pointwise scheduler does not hoist BN-inference channel broadcasts into a channel-blocked affine epilogue while preserving the explicit bf16 rounding before ReLU; the fix is NEW_PATTERN: add a guarded BN-inference affine-ReLU pointwise template that computes per-channel normalization scalars once per tile for dense NCHW and channels-last layouts."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bn_relu_flat_kernel(
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
        c_offsets = offsets % C
    else:
        c_offsets = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)

    invstd = 1.0 / tl.sqrt(var + 0.001)
    y = (x - mean) * invstd * weight + bias
    y_bf16 = y.to(tl.bfloat16)
    out = tl.where(y_bf16 != y_bf16, y_bf16, tl.maximum(y_bf16, 0.0))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _bn_relu_nchw_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)
    hw_block = tl.program_id(2)

    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_mask = c_offsets < C
    hw_mask = hw_offsets < HW
    mask = c_mask[:, None] & hw_mask[None, :]

    offsets = n * C * HW + c_offsets[:, None] * HW + hw_offsets[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    invstd = 1.0 / tl.sqrt(var + 0.001)
    y = (x - mean[:, None]) * invstd[:, None] * weight[:, None] + bias[:, None]
    y_bf16 = y.to(tl.bfloat16)
    out = tl.where(y_bf16 != y_bf16, y_bf16, tl.maximum(y_bf16, 0.0))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _bn_relu_channels_last_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    hw_block = tl.program_id(1)
    c_block = tl.program_id(2)

    hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_mask = hw_offsets < HW
    c_mask = c_offsets < C
    mask = hw_mask[:, None] & c_mask[None, :]

    offsets = n * C * HW + hw_offsets[:, None] * C + c_offsets[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    invstd = 1.0 / tl.sqrt(var + 0.001)
    y = (x - mean[None, :]) * invstd[None, :] * weight[None, :] + bias[None, :]
    y_bf16 = y.to(tl.bfloat16)
    out = tl.where(y_bf16 != y_bf16, y_bf16, tl.maximum(y_bf16, 0.0))
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="aeef24a2", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="d0015d69", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="79f50b00", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="45fcf39e", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="6b130750", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="466338df", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="2eaf76ca", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="5ecba4cf", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="2869e39e", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="8a535339", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="f868ccff", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="2b758769", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="a034af1b", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="4c825568", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="3d7c0f11", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="76836be5", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="37cf4567", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="09d7b0e3", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="ff6665c7", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="2d204c68", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="0a96753f", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="c7c6ed1f", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="e7ac2695", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="5ddd3fd5", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="b88e53cc", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="9c97edfa", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="55c49b50", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="9ffe1d7a", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="d0456ef8", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="dc068d28", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="23b42229", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="8273eec8", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="fe78a763", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="fb1609c3", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="3a805d91", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="f9b980a0", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="9a46a7c3", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="29619cda", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="49317bb2", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="15c8ba03", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="5536024c", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="706b5a8b", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="0c615bbc", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="86d8b817", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="a01c3bbe", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="ba61c626", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="bc2770a8", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="9e8f8bc2", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="8004dcee", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="6039b181", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="d438d876", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="b6537524", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="cc855e4e", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="27be3f93", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="a3bf2c8d", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="0fe88f70", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="6558df98", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="793856d7", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="c2f1485a", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="78e9098c", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="4fbb328e", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="c1f1985a", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="b83c3106", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="82781865", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="3a0fa905", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="e3b17b69", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="d6d70882", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="c9277843", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="6cc76740", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="08df9f87", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="a952c87d", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="51b8f364", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="e78adaa5", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="09db272d", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="1f81c187", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="b5cedc58", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="c78ca241", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="11ab0e65", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="4194c732", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="05028dd8", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="4f218228", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="0ce7c2f4", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="535abe76", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="3f17dc95", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="d957323e", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="7fd9f949", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="2b7370d5", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="6ff82f3b", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="f50736b8", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="ad2ff10f", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="7ebdbaf5", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="8380c440", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="e3fcf83d", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="20f76211", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="2ff0039c", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="93683cfe", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="6b05bfa0", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="6a1621d9", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="f7d56653", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="c7ca5fcc", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="eff1e63e", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="18ebcf69", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="13e72620", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="b30eafb6", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="f3036c42", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="eb73312d", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="9a05b224", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="c356b4b1", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="a8109c98", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="a9c4f100", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="7e261bd0", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="fe80f1ea", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="217063c8", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="13bff0d2", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="23eaa07b", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="dd92c2d0", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="357d64ea", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="e84dbb88", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="839e83c4", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="608e0114", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="a2b8c17d", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int = 16,
    BLOCK_HW: int = 64,
    BLOCK_N: int = 256,
    num_warps: int = 4,
):
    mean, x, var, weight, bias = inputs
    _, channels, height, width = x.shape
    hw = height * width
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    if x.stride(1) == 1:
        grid = (triton.cdiv(x.numel(), BLOCK_N),)
        _bn_relu_flat_kernel[grid](
            mean,
            x,
            var,
            weight,
            bias,
            out,
            TOTAL=x.numel(),
            C=channels,
            HW=hw,
            CHANNELS_LAST=True,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
    else:
        grid = (
            x.shape[0],
            triton.cdiv(channels, BLOCK_C),
            triton.cdiv(hw, BLOCK_HW),
        )
        _bn_relu_nchw_kernel[grid](
            mean,
            x,
            var,
            weight,
            bias,
            out,
            C=channels,
            HW=hw,
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            num_warps=num_warps,
        )
    return out
