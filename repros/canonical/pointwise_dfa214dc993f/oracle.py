"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 SqueezeNet `relu`, channel-`cat`, and 3x3 stride-2 ceil-mode maxpool-values scope in one Triton stencil kernel, writing only the returned contiguous pooled values tensor and avoiding the unreturned ReLU, concat, and maxpool-offset materializations; Inductor lowers the decomposed branch ReLUs, channel concatenation, and pooling stencil through generic scheduled work around materialized intermediates; the fix is SCHEDULER_FUSION: teach the scheduler to sink equal-shape branch producers and virtual channel concatenations into a following pooling stencil while preserving bf16 ReLU and max comparison semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_cat_maxpool_values_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    ow = offsets % out_width
    tmp = offsets // out_width
    oh = tmp % out_height
    tmp = tmp // out_height
    out_channel = tmp % (channels * 2)
    batch = tmp // (channels * 2)

    use_lhs = out_channel < channels
    channel = tl.where(use_lhs, out_channel, out_channel - channels)
    input_base = batch * channels * height * width + channel * height * width
    input_base += (oh * 2) * width + ow * 2

    neg_inf = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    best = neg_inf
    zero = tl.full((BLOCK_SIZE,), 0.0, tl.float32)

    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            in_offsets = input_base + kh * width + kw
            ptrs = tl.where(use_lhs, lhs_ptr + in_offsets, rhs_ptr + in_offsets)
            vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            vals = tl.where((vals > zero) | (vals != vals), vals, zero)
            better = (vals > best) | (vals != vals)
            best = tl.where(better, vals, best)

    tl.store(out_ptr + offsets, best, mask=mask)


def _launch(
    inputs,
    *,
    CHANNELS,
    HEIGHT,
    WIDTH,
    OUT_HEIGHT,
    OUT_WIDTH,
    BLOCK_SIZE,
    num_warps,
):
    lhs, rhs, _kernel_size, _stride = inputs
    batch = lhs.shape[0]
    out = torch.empty_strided(
        (batch, CHANNELS * 2, OUT_HEIGHT, OUT_WIDTH),
        (CHANNELS * 2 * OUT_HEIGHT * OUT_WIDTH, OUT_HEIGHT * OUT_WIDTH, OUT_WIDTH, 1),
        device=lhs.device,
        dtype=lhs.dtype,
    )
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _relu_cat_maxpool_values_kernel[grid](
        lhs,
        rhs,
        out,
        n_elements,
        channels=CHANNELS,
        height=HEIGHT,
        width=WIDTH,
        out_height=OUT_HEIGHT,
        out_width=OUT_WIDTH,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="B200", point="86725088", CHANNELS=128, HEIGHT=27, WIDTH=27, OUT_HEIGHT=13, OUT_WIDTH=13, BLOCK_SIZE=128, num_warps=4)
@oracle_impl(hardware="B200", point="5a0347c9", CHANNELS=64, HEIGHT=55, WIDTH=55, OUT_HEIGHT=27, OUT_WIDTH=27, BLOCK_SIZE=128, num_warps=4)
def oracle_forward(inputs, *, CHANNELS, HEIGHT, WIDTH, OUT_HEIGHT, OUT_WIDTH, BLOCK_SIZE, num_warps):
    return _launch(
        inputs,
        CHANNELS=CHANNELS,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        OUT_HEIGHT=OUT_HEIGHT,
        OUT_WIDTH=OUT_WIDTH,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
