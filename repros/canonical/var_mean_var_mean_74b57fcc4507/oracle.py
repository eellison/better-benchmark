"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet dual training-BatchNorm channel-shuffle scope, including both bf16-to-fp32 population var_mean reductions over N,H,W, eps=1e-5 rsqrt side outputs, four mutable running-stat copy_ updates with the captured correction literal, both affine bf16/ReLU epilogues, the virtual cat/view/permute/clone/view channel shuffle, and the returned split aliases from one shared backing tensor, whereas Inductor lowers the sibling BN reductions, mutable running-stat epilogues, concat, permutation clone, and split as generic reduction and layout work; Inductor cannot do this today because its scheduler does not co-schedule paired training-BN templates with side-effect outputs while sinking a fixed ShuffleNet channel shuffle into the output layout; the fix is SCHEDULER_FUSION: extend the BN-training/layout scheduler to group sibling channel statistics, emit running-stat side effects, and write the shuffled split backing storage directly while preserving bf16 cast boundaries and aliases."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.00000996502277


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
def _dual_partial_sums_kernel(
    x0_ptr,
    x1_ptr,
    partial_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    x0_stride_n: tl.constexpr,
    x0_stride_c: tl.constexpr,
    x0_stride_h: tl.constexpr,
    x0_stride_w: tl.constexpr,
    x1_stride_n: tl.constexpr,
    x1_stride_c: tl.constexpr,
    x1_stride_h: tl.constexpr,
    x1_stride_w: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunk = tl.program_id(1)
    r_offsets = chunk * BLOCK_R + tl.arange(0, BLOCK_R)

    r_mask = r_offsets < elements_per_channel
    c_mask = c_offsets < channels
    n_idx = r_offsets // hw_size
    hw = r_offsets - n_idx * hw_size
    h_idx = hw // width
    w_idx = hw - h_idx * width
    mask = c_mask[:, None] & r_mask[None, :]

    x0_offsets = (
        n_idx[None, :] * x0_stride_n
        + c_offsets[:, None] * x0_stride_c
        + h_idx[None, :] * x0_stride_h
        + w_idx[None, :] * x0_stride_w
    )
    x1_offsets = (
        n_idx[None, :] * x1_stride_n
        + c_offsets[:, None] * x1_stride_c
        + h_idx[None, :] * x1_stride_h
        + w_idx[None, :] * x1_stride_w
    )

    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)
    sum0 = tl.sum(x0, axis=1)
    sum1 = tl.sum(x1, axis=1)
    sumsq0 = tl.sum(_f32_mul(x0, x0), axis=1)
    sumsq1 = tl.sum(_f32_mul(x1, x1), axis=1)

    out_offsets = chunk * channels + c_offsets
    plane_stride = num_chunks * channels
    tl.store(partial_ptr + out_offsets, sum0, mask=c_mask)
    tl.store(partial_ptr + plane_stride + out_offsets, sumsq0, mask=c_mask)
    tl.store(partial_ptr + 2 * plane_stride + out_offsets, sum1, mask=c_mask)
    tl.store(partial_ptr + 3 * plane_stride + out_offsets, sumsq1, mask=c_mask)


@triton.jit
def _dual_finalize_sums_kernel(
    partial_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    mean0_ptr,
    invstd0_ptr,
    mean1_ptr,
    invstd1_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_chunks: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (c_offsets[:, None] < channels) & (chunks[None, :] < num_chunks)
    offsets = chunks[None, :] * channels + c_offsets[:, None]
    plane_stride = num_chunks * channels

    sum0 = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0), axis=1)
    sumsq0 = tl.sum(tl.load(partial_ptr + plane_stride + offsets, mask=mask, other=0.0), axis=1)
    sum1 = tl.sum(tl.load(partial_ptr + 2 * plane_stride + offsets, mask=mask, other=0.0), axis=1)
    sumsq1 = tl.sum(tl.load(partial_ptr + 3 * plane_stride + offsets, mask=mask, other=0.0), axis=1)

    inv_count = 1.0 / elements_per_channel
    mean0 = _f32_mul(sum0, inv_count)
    mean1 = _f32_mul(sum1, inv_count)
    var0 = _f32_sub(_f32_mul(sumsq0, inv_count), _f32_mul(mean0, mean0))
    var1 = _f32_sub(_f32_mul(sumsq1, inv_count), _f32_mul(mean1, mean1))
    var0 = tl.where(var0 < 0.0, 0.0, var0)
    var1 = tl.where(var1 < 0.0, 0.0, var1)
    invstd0 = libdevice.rsqrt(_f32_add(var0, eps))
    invstd1 = libdevice.rsqrt(_f32_add(var1, eps))

    channel_mask = c_offsets < channels
    old_mean0 = tl.load(running_mean0_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)

    new_mean0 = _f32_add(_f32_mul(mean0, momentum), _f32_mul(old_mean0, 1.0 - momentum))
    new_var0 = _f32_add(
        _f32_mul(_f32_mul(var0, running_var_correction), momentum),
        _f32_mul(old_var0, 1.0 - momentum),
    )
    new_mean1 = _f32_add(_f32_mul(mean1, momentum), _f32_mul(old_mean1, 1.0 - momentum))
    new_var1 = _f32_add(
        _f32_mul(_f32_mul(var1, running_var_correction), momentum),
        _f32_mul(old_var1, 1.0 - momentum),
    )

    tl.store(mean0_ptr + c_offsets, mean0, mask=channel_mask)
    tl.store(invstd0_ptr + c_offsets, invstd0, mask=channel_mask)
    tl.store(mean1_ptr + c_offsets, mean1, mask=channel_mask)
    tl.store(invstd1_ptr + c_offsets, invstd1, mask=channel_mask)
    tl.store(running_mean0_ptr + c_offsets, new_mean0, mask=channel_mask)
    tl.store(running_var0_ptr + c_offsets, new_var0, mask=channel_mask)
    tl.store(running_mean1_ptr + c_offsets, new_mean1, mask=channel_mask)
    tl.store(running_var1_ptr + c_offsets, new_var1, mask=channel_mask)


@triton.jit
def _dual_bn_shuffle_kernel(
    x0_ptr,
    x1_ptr,
    weight0_ptr,
    bias0_ptr,
    weight1_ptr,
    bias1_ptr,
    mean0_ptr,
    invstd0_ptr,
    mean1_ptr,
    invstd1_ptr,
    out_ptr,
    total_elements: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    out_channels: tl.constexpr,
    x0_stride_n: tl.constexpr,
    x0_stride_c: tl.constexpr,
    x0_stride_h: tl.constexpr,
    x0_stride_w: tl.constexpr,
    x1_stride_n: tl.constexpr,
    x1_stride_c: tl.constexpr,
    x1_stride_h: tl.constexpr,
    x1_stride_w: tl.constexpr,
    width: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < total_elements
    hw = offsets - (offsets // hw_size) * hw_size
    channel = (offsets // hw_size) - ((offsets // hw_size) // channels) * channels
    n_idx = offsets // (channels * hw_size)
    h_idx = hw // width
    w_idx = hw - h_idx * width

    x0_offsets = (
        n_idx * x0_stride_n
        + channel * x0_stride_c
        + h_idx * x0_stride_h
        + w_idx * x0_stride_w
    )
    x1_offsets = (
        n_idx * x1_stride_n
        + channel * x1_stride_c
        + h_idx * x1_stride_h
        + w_idx * x1_stride_w
    )

    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)
    mean0 = tl.load(mean0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd0 = tl.load(invstd0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd1 = tl.load(invstd1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    y0 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x0, mean0), invstd0), weight0), bias0)
    y1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x1, mean1), invstd1), weight1), bias1)
    rounded0 = y0.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    rounded1 = y1.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    relu0 = tl.where((rounded0 > 0.0) | (rounded0 != rounded0), rounded0, 0.0)
    relu1 = tl.where((rounded1 > 0.0) | (rounded1 != rounded1), rounded1, 0.0)

    out_base = n_idx * out_channels * hw_size + channel * 2 * hw_size + hw
    tl.store(out_ptr + out_base, relu0.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
    tl.store(out_ptr + out_base + hw_size, relu1.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# ce5d17b1: (T([128,58,28,28], bf16), T([58], f32), ..., T([128,58,28,28], bf16), ...)
@oracle_impl(
    hardware="B200",
    point="ce5d17b1",
    BLOCK_R=4096,
    BLOCK_C=8,
    BLOCK_E=1024,
    num_warps_stats=8,
    num_warps_final=1,
    num_warps_act=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_E: int,
    num_warps_stats: int,
    num_warps_final: int,
    num_warps_act: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        _shape_param_0,
        _shape_param_1,
    ) = inputs
    del _shape_param_0, _shape_param_1

    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw_size = height * width
    elements_per_channel = batch * hw_size
    total_elements = batch * channels * hw_size
    out_channels = channels * 2
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    mean0 = torch.empty_strided((1, channels, 1, 1), (channels, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    invstd0 = torch.empty_strided((1, channels, 1, 1), (channels, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    mean1 = torch.empty_strided((1, channels, 1, 1), (channels, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    invstd1 = torch.empty_strided((1, channels, 1, 1), (channels, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(
        (batch, out_channels, height, width),
        (out_channels * hw_size, hw_size, width, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (4, num_chunks, channels),
        (num_chunks * channels, channels, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _dual_partial_sums_kernel[(triton.cdiv(channels, BLOCK_C), num_chunks)](
        arg0_1,
        arg5_1,
        partial,
        channels=channels,
        width=width,
        hw_size=hw_size,
        elements_per_channel=elements_per_channel,
        x0_stride_n=int(arg0_1.stride(0)),
        x0_stride_c=int(arg0_1.stride(1)),
        x0_stride_h=int(arg0_1.stride(2)),
        x0_stride_w=int(arg0_1.stride(3)),
        x1_stride_n=int(arg5_1.stride(0)),
        x1_stride_c=int(arg5_1.stride(1)),
        x1_stride_h=int(arg5_1.stride(2)),
        x1_stride_w=int(arg5_1.stride(3)),
        num_chunks=num_chunks,
        BLOCK_C=BLOCK_C,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _dual_finalize_sums_kernel[(triton.cdiv(channels, BLOCK_C),)](
        partial,
        arg1_1,
        arg2_1,
        arg6_1,
        arg7_1,
        mean0,
        invstd0,
        mean1,
        invstd1,
        channels=channels,
        elements_per_channel=elements_per_channel,
        num_chunks=num_chunks,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_final,
        num_stages=3,
    )
    _dual_bn_shuffle_kernel[(triton.cdiv(total_elements, BLOCK_E),)](
        arg0_1,
        arg5_1,
        arg3_1,
        arg4_1,
        arg8_1,
        arg9_1,
        mean0,
        invstd0,
        mean1,
        invstd1,
        out,
        total_elements=total_elements,
        channels=channels,
        hw_size=hw_size,
        out_channels=out_channels,
        x0_stride_n=int(arg0_1.stride(0)),
        x0_stride_c=int(arg0_1.stride(1)),
        x0_stride_h=int(arg0_1.stride(2)),
        x0_stride_w=int(arg0_1.stride(3)),
        x1_stride_n=int(arg5_1.stride(0)),
        x1_stride_c=int(arg5_1.stride(1)),
        x1_stride_h=int(arg5_1.stride(2)),
        x1_stride_w=int(arg5_1.stride(3)),
        width=width,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps_act,
        num_stages=3,
    )

    return (
        mean0,
        invstd0,
        mean1,
        invstd1,
        out[:, :channels, :, :],
        out[:, channels:, :, :],
        arg1_1,
        arg2_1,
        arg6_1,
        arg7_1,
    )
