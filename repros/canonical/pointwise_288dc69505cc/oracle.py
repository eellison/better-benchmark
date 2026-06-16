"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete NFNet bf16 exact-erf GELU scale and right/bottom constant_pad_nd scope with per-point layout-specialized channels-last Triton schedules, using a direct output-space kernel for the small point and split interior plus static-pad-fringe kernels for the larger points, whereas Inductor lowers the same graph as one generic output-space pointwise/pad kernel with masked pad lanes; Inductor cannot do this today because its pointwise/pad codegen has no guarded NFNet exact-GELU-scale-into-static-pad template that selects between direct output traversal and decomposed interior/fringe schedules while preserving the generated fp32 math and final bf16 store; the fix is NEW_PATTERN: add a guarded channels-last exact-GELU-scale plus static-pad lowering that emits the best layout-specialized schedule for each static shape."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gelu_scale_pad_output_nhwc_kernel(
    input_ptr,
    output_ptr,
    total_out: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_out

    out_h: tl.constexpr = height + 1
    out_w: tl.constexpr = width + 1
    out_row: tl.constexpr = out_w * channels
    in_row: tl.constexpr = width * channels
    in_plane: tl.constexpr = height * width * channels

    h = (offsets // out_row) % out_h
    w = (offsets // channels) % out_w
    n = offsets // (out_h * out_row)
    row_offset = offsets % out_row
    in_bounds = mask & (h < height) & (w < width)
    in_offsets = n * in_plane + h * in_row + row_offset

    x = tl.load(input_ptr + in_offsets, mask=in_bounds, other=0.0).to(tl.float32)
    half = x * 0.5
    erf_arg = x * 0.7071067811865476
    gelu = half * (libdevice.erf(erf_arg) + 1.0)
    scaled = gelu * 1.7015043497085571
    out_value = tl.where(in_bounds, scaled, 0.0)
    tl.store(output_ptr + offsets, out_value, mask=mask)


@triton.jit
def _gelu_scale_pad_interior_nhwc_kernel(
    input_ptr,
    output_ptr,
    total_in: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_in

    c = offsets % channels
    tmp = offsets // channels
    w = tmp % width
    tmp = tmp // width
    h = tmp % height
    n = tmp // height

    out_h: tl.constexpr = height + 1
    out_w: tl.constexpr = width + 1
    out_offsets = ((n * out_h + h) * out_w + w) * channels + c

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    half = x * 0.5
    erf_arg = x * 0.7071067811865476
    gelu = half * (libdevice.erf(erf_arg) + 1.0)
    scaled = gelu * 1.7015043497085571
    tl.store(output_ptr + out_offsets, scaled, mask=mask)


@triton.jit
def _zero_pad_fringe_nhwc_kernel(
    output_ptr,
    total_pad: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_pad

    c = offsets % channels
    tmp = offsets // channels
    pos = tmp % (height + width + 1)
    n = tmp // (height + width + 1)

    out_h: tl.constexpr = height + 1
    out_w: tl.constexpr = width + 1
    right_col = pos < height
    h = tl.where(right_col, pos, height)
    w = tl.where(right_col, width, pos - height)
    out_offsets = ((n * out_h + h) * out_w + w) * channels + c
    tl.store(output_ptr + out_offsets, 0.0, mask=mask)


@oracle_impl(hardware="B200", point="192c8e99", BLOCK=1024, PAD_BLOCK=512, SPLIT=False, num_warps=4)
@oracle_impl(hardware="B200", point="94a90df4", BLOCK=512, PAD_BLOCK=512, SPLIT=True, num_warps=4)
@oracle_impl(hardware="B200", point="4ca56f88", BLOCK=512, PAD_BLOCK=512, SPLIT=True, num_warps=4)
@oracle_impl(hardware="B200", point="a776dd34", BLOCK=512, PAD_BLOCK=512, SPLIT=True, num_warps=4)
@oracle_impl(hardware="B200", point="95428590", BLOCK=512, PAD_BLOCK=512, SPLIT=True, num_warps=4)
@oracle_impl(hardware="B200", point="7b76ed5c", BLOCK=512, PAD_BLOCK=512, SPLIT=True, num_warps=4)
@oracle_impl(hardware="B200", point="f0a667e3", BLOCK=512, PAD_BLOCK=512, SPLIT=True, num_warps=4)
@oracle_impl(hardware="B200", point="c5d259c9", BLOCK=512, PAD_BLOCK=512, SPLIT=True, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, PAD_BLOCK: int, SPLIT: bool, num_warps: int):
    arg0_1 = inputs[0]
    batch, channels, height, width = arg0_1.shape
    out_h = height + 1
    out_w = width + 1
    out = torch.empty_strided(
        (batch, channels, out_h, out_w),
        (channels * out_h * out_w, 1, out_w * channels, channels),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    total_out = out.numel()
    if not SPLIT:
        _gelu_scale_pad_output_nhwc_kernel[(triton.cdiv(total_out, BLOCK),)](
            arg0_1,
            out,
            total_out=total_out,
            channels=channels,
            height=height,
            width=width,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

    total_in = arg0_1.numel()
    _gelu_scale_pad_interior_nhwc_kernel[(triton.cdiv(total_in, BLOCK),)](
        arg0_1,
        out,
        total_in=total_in,
        channels=channels,
        height=height,
        width=width,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )

    total_pad = batch * channels * (height + width + 1)
    _zero_pad_fringe_nhwc_kernel[(triton.cdiv(total_pad, PAD_BLOCK),)](
        out,
        total_pad=total_pad,
        channels=channels,
        height=height,
        width=width,
        BLOCK=PAD_BLOCK,
        num_warps=num_warps,
    )
    return out
