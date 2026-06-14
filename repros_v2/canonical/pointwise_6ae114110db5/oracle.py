"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete CycleGAN bf16 reflected-padding scope in one Triton gather kernel, including the bf16-to-fp32 value-preserving input cast, both fixed reflection index expressions, the eliminated intermediate `[1,3,262,256]` gather, and the final bf16 `[1,3,262,262]` output; Inductor schedules the two `aten._unsafe_index` consumers and dtype boundary as generic pointwise/indexing regions, so it materializes the first reflected layout instead of writing the final reflected layout directly; the fix is SCHEDULER_FUSION: teach pointwise/indexing scheduling to fold static reflection-index consumers and dtype-boundary casts into one direct final-layout store."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _reflect_pad_kernel(
    x_ptr,
    out_ptr,
    in_stride_n: tl.constexpr,
    in_stride_c: tl.constexpr,
    in_stride_h: tl.constexpr,
    in_stride_w: tl.constexpr,
    total: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_h_size: tl.constexpr,
    out_w_size: tl.constexpr,
    pad: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < total

    ow = offsets % out_w_size
    oh = (offsets // out_w_size) % out_h_size
    c = (offsets // (out_h_size * out_w_size)) % channels
    n = offsets // (channels * out_h_size * out_w_size)

    h_shifted = oh - pad
    h_abs = tl.where(h_shifted < 0, -h_shifted, h_shifted)
    h_reflected = tl.where(h_abs <= height - 1, h_abs, 2 * (height - 1) - h_abs)
    w_shifted = ow - pad
    w_abs = tl.where(w_shifted < 0, -w_shifted, w_shifted)
    w_reflected = tl.where(w_abs <= width - 1, w_abs, 2 * (width - 1) - w_abs)

    in_offsets = (
        n * in_stride_n
        + c * in_stride_c
        + h_reflected * in_stride_h
        + w_reflected * in_stride_w
    )
    vals = tl.load(x_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, vals.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="3fee83c6", BLOCK_N=256, num_warps=2, num_stages=3)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
    (x,) = inputs
    channels = 3
    height = 256
    width = 256
    pad = 3
    out_h = height + pad + pad
    out_w = width + pad + pad
    total = channels * out_h * out_w
    out = torch.empty_strided(
        (1, channels, out_h, out_w),
        (channels * out_h * out_w, out_h * out_w, out_w, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _reflect_pad_kernel[(triton.cdiv(total, BLOCK_N),)](
        x,
        out,
        in_stride_n=x.stride(0),
        in_stride_c=x.stride(1),
        in_stride_h=x.stride(2),
        in_stride_w=x.stride(3),
        total=total,
        channels=channels,
        height=height,
        width=width,
        out_h_size=out_h,
        out_w_size=out_w,
        pad=pad,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
