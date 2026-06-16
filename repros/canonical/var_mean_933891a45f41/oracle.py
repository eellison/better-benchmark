"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete CycleGAN bf16 instance-normalization, bf16 residual add, returned unpadded add tensor, and two-axis reflected unsafe-index output with one channel-statistics kernel plus one parallel epilogue/reflection kernel, whereas Inductor schedules the fp32 channel statistics, bf16 normalization/add epilogue, returned producer, and reflected indexing as generic materialized regions; Inductor cannot do this today because its normalization scheduler does not treat fixed reflected-index consumers and live returned bf16 epilogues as direct stores from the channel-normalization tile; the fix is SCHEDULER_FUSION: extend norm-template fusion to sink static reflection layout consumers while preserving the returned bf16 producer and exact cast boundaries."""

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
def _channel_stats_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    HW: tl.constexpr,
    STAT_BLOCK: tl.constexpr,
):
    channel = tl.program_id(0)
    stat_offsets = tl.arange(0, STAT_BLOCK)
    x = tl.load(x_ptr + channel * HW + stat_offsets).to(tl.float32)

    mean = tl.sum(x, axis=0) / HW
    centered = _f32_sub(x, mean)
    variance = tl.sum(_f32_mul(centered, centered), axis=0) / HW
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))
    tl.store(mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)


@triton.jit
def _add_and_reflect_kernel(
    x_ptr,
    residual_ptr,
    mean_ptr,
    invstd_ptr,
    add_out_ptr,
    reflected_out_ptr,
    TOTAL_IN: tl.constexpr,
    TOTAL_OUT: tl.constexpr,
    HW: tl.constexpr,
    OUT_HW: tl.constexpr,
    PADDED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    in_mask = offsets < TOTAL_IN
    in_channel = offsets // HW
    y_in = tl.load(x_ptr + offsets, mask=in_mask, other=0.0).to(tl.float32)
    residual_in = tl.load(residual_ptr + offsets, mask=in_mask, other=0.0).to(tl.float32)
    mean_in = tl.load(mean_ptr + in_channel, mask=in_mask, other=0.0).to(tl.float32)
    invstd_in = tl.load(invstd_ptr + in_channel, mask=in_mask, other=0.0).to(tl.float32)
    norm_in = _f32_mul(_f32_sub(y_in, mean_in), invstd_in)
    rounded_in = norm_in.to(tl.bfloat16).to(tl.float32)
    added_in = _f32_add(residual_in, rounded_in).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, added_in, mask=in_mask)

    out_mask = offsets < TOTAL_OUT
    out_channel = offsets // OUT_HW
    spatial = offsets - out_channel * OUT_HW
    oh = spatial // PADDED
    ow = spatial - oh * PADDED

    ih = tl.where(oh == 0, 1, tl.where(oh == PADDED - 1, WIDTH - 2, oh - 1))
    iw = tl.where(ow == 0, 1, tl.where(ow == PADDED - 1, WIDTH - 2, ow - 1))
    reflected_input_offsets = out_channel * HW + ih * WIDTH + iw

    y_out = tl.load(x_ptr + reflected_input_offsets, mask=out_mask, other=0.0).to(tl.float32)
    residual_out = tl.load(
        residual_ptr + reflected_input_offsets,
        mask=out_mask,
        other=0.0,
    ).to(tl.float32)
    mean_out = tl.load(mean_ptr + out_channel, mask=out_mask, other=0.0).to(tl.float32)
    invstd_out = tl.load(invstd_ptr + out_channel, mask=out_mask, other=0.0).to(tl.float32)
    norm_out = _f32_mul(_f32_sub(y_out, mean_out), invstd_out)
    rounded_out = norm_out.to(tl.bfloat16).to(tl.float32)
    added_out = _f32_add(residual_out, rounded_out).to(tl.bfloat16)
    tl.store(reflected_out_ptr + offsets, added_out, mask=out_mask)


# 50d031e1: (T([1,256,64,64], bf16), T([1,256,64,64], bf16))
@oracle_impl(hardware="B200", point="50d031e1", STAT_BLOCK=4096, BLOCK=1024, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, STAT_BLOCK: int, BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1 = inputs
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    padded = width + 2
    out_hw = padded * padded
    total_in = channels * hw
    total_out = channels * out_hw

    mean = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    invstd = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (1, channels, height, width),
        (channels * hw, hw, width, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    reflected_out = torch.empty_strided(
        (1, channels, padded, padded),
        (channels * out_hw, out_hw, padded, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _channel_stats_kernel[(channels,)](
        arg0_1,
        mean,
        invstd,
        HW=hw,
        STAT_BLOCK=STAT_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _add_and_reflect_kernel[(triton.cdiv(total_out, BLOCK),)](
        arg0_1,
        arg1_1,
        mean,
        invstd,
        add_out,
        reflected_out,
        TOTAL_IN=total_in,
        TOTAL_OUT=total_out,
        HW=hw,
        OUT_HW=out_hw,
        PADDED=padded,
        WIDTH=width,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=num_stages,
    )
    return add_out, reflected_out
