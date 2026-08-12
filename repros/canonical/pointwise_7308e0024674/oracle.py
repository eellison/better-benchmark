"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 dual BN-affine, bf16 cast, NaN-preserving ReLU, channel cat, view, permute, contiguous clone, final view, and split-return scope by writing both producer branches directly into the final channel-shuffled backing storage and returning the exact two split views, whereas Inductor currently fuses each branch producer but still materializes the unshuffled cat buffer and then launches generic layout work for the view/permute/clone/split chain; Inductor cannot do this today because scheduler fusion does not propagate the consumer layout transform and split offsets back into the producer store indexing across reshape/permute/clone-only layout chains; the fix is SCHEDULER_FUSION: teach scheduler/codegen to sink final layout indexing through reshape/permute/clone/split chains and store fused pointwise producers directly in the consumer layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 64
CHANNELS = 58
HEIGHT = 28
WIDTH = 28
HW = HEIGHT * WIDTH
OUT_CHANNELS = 2 * CHANNELS
BRANCH_NUMEL = BATCH * CHANNELS * HW
EPS = 1.0e-5


@triton.jit
def _relu_after_bf16(x):
    y = x.to(tl.bfloat16)
    return tl.where((y > 0.0) | (y != y), y, 0.0).to(tl.bfloat16)


@triton.jit
def _shuffle_bn_relu_kernel(
    mean_a_ptr,
    conv_a_ptr,
    var_a_ptr,
    weight_a_ptr,
    bias_a_ptr,
    mean_b_ptr,
    conv_b_ptr,
    var_b_ptr,
    weight_b_ptr,
    bias_b_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < (64 * 58 * 784)

    hw = offsets % 784
    channel = (offsets // 784) % 58
    batch = offsets // (58 * 784)
    src_offsets = batch * (58 * 784) + channel * 784 + hw

    a_x = tl.load(conv_a_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    a_mean = tl.load(mean_a_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    a_var = tl.load(var_a_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    a_weight = tl.load(weight_a_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    a_bias = tl.load(bias_a_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    a_norm = (a_x - a_mean) * (1.0 / tl.sqrt(a_var + 1.0e-5))
    a_out = _relu_after_bf16(a_norm * a_weight + a_bias)

    b_x = tl.load(conv_b_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    b_mean = tl.load(mean_b_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.load(var_b_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    b_weight = tl.load(weight_b_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    b_bias = tl.load(bias_b_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    b_norm = (b_x - b_mean) * (1.0 / tl.sqrt(b_var + 1.0e-5))
    b_out = _relu_after_bf16(b_norm * b_weight + b_bias)

    out_offsets = batch * (116 * 784) + (2 * channel) * 784 + hw
    tl.store(out_ptr + out_offsets, a_out, mask=mask)
    tl.store(out_ptr + out_offsets + 784, b_out, mask=mask)


@oracle_impl(hardware="B200", point="cff026d0", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (
        mean_a,
        conv_a,
        var_a,
        weight_a,
        bias_a,
        mean_b,
        conv_b,
        var_b,
        weight_b,
        bias_b,
        _shape0,
        _shape1,
    ) = inputs
    shuffled = torch.empty_strided(
        (BATCH, OUT_CHANNELS, HEIGHT, WIDTH),
        (OUT_CHANNELS * HW, HW, WIDTH, 1),
        device=conv_a.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(BRANCH_NUMEL, BLOCK),)
    _shuffle_bn_relu_kernel[grid](
        mean_a,
        conv_a,
        var_a,
        weight_a,
        bias_a,
        mean_b,
        conv_b,
        var_b,
        weight_b,
        bias_b,
        shuffled,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return shuffled[:, :CHANNELS, :, :], shuffled[:, CHANNELS:, :, :]
