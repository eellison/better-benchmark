"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete channels-last right-and-bottom zero `constant_pad_nd` plus f32-to-bf16 cast scope by directly materializing the fresh padded bf16 output, copying and rounding every interior input element and zero-filling only the narrow pad fringe, whereas Inductor already lowers this isolated pad/cast graph to comparable mandatory read/write work with generic pad indexing; Inductor cannot materially improve this local repro today because the exact contract still requires a fresh channels-last padded allocation, reading every interior f32 element, bf16-rounding each copied value, and writing the zero pad fringe with no producer or consumer fusion left to remove; the fix is BANDWIDTH_BOUND: record this as an at-floor pad-copy case unless broader pointwise pad/cast memory codegen or launch-overhead work moves both implementations together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _pad_cast_output_nhwc_kernel(
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

    values = tl.load(input_ptr + in_offsets, mask=in_bounds, other=0.0).to(tl.float32)
    values = tl.where(in_bounds, values, 0.0)
    tl.store(output_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="70bbd2d7", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="45c33f0e", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
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
    _pad_cast_output_nhwc_kernel[(triton.cdiv(total_out, BLOCK),)](
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
