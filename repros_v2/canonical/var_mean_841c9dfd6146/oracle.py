"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Visformer training-BatchNorm add scope, including the fp32+bf16 add producer returned as output, population var_mean over N/H/W, saved invstd, affine bf16 cast, centered fp32 output, and both mutable running-stat copy_ aliases, whereas Inductor lowers the decomposed add, norm-template statistics, running-stat updates, and sibling materialized consumers through generic schedules; Inductor cannot do this today because the training-normalization scheduler does not preserve a cheap mixed-dtype add producer together with multiple post-stat full-tensor consumers and mutable side outputs in one compact channels-last plan; the fix is SCHEDULER_FUSION: extend the BN-training template to fuse structured add producers, saved-stat returns, running-stat aliases, and affine/centered epilogues under one benchmark-gated schedule."""

import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl

triton_helpers.set_driver_to_gpu()


def _device_properties():
    return DeviceProperties.create(torch.device("cuda", torch.cuda.current_device()))


def _triton_meta(signature):
    return {
        "signature": signature,
        "device": _device_properties(),
        "constants": {},
        "configs": [{}],
    }


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


@triton_heuristics.reduction(
    size_hints={"x": 65536, "r0_": 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta=_triton_meta(
        {
            "x0_ptr": "*fp32",
            "x1_ptr": "*bf16",
            "partial_ptr": "*fp32",
            "xnumel": "i32",
            "r0_numel": "i32",
            "XBLOCK": "constexpr",
            "R0_BLOCK": "constexpr",
        }
    ),
    inductor_meta={
        "grid_type": "Grid1D",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "autotune_pointwise": True,
        "coordinate_descent_tuning": True,
    },
)
@triton.jit
def _d9_partial_add_welford_kernel(
    x0_ptr,
    x1_ptr,
    partial_ptr,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xnumel = 37632
    r0_numel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    channel = xindex % 768
    hw = xindex // 768

    mean_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    m2_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    weight_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        offsets = channel + 768 * r0_index + 98304 * hw
        x0 = tl.load(x0_ptr + offsets, r0_mask & xmask, eviction_policy="evict_first", other=0.0)
        x1 = tl.load(x1_ptr + offsets, r0_mask & xmask, eviction_policy="evict_first", other=0.0).to(tl.float32)
        vals = _f32_add(x0.to(tl.float32), x1)
        vals = tl.broadcast_to(vals, [XBLOCK, R0_BLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            vals,
            mean_acc,
            m2_acc,
            weight_acc,
            r0_offset == 0,
        )
        mean_acc = tl.where(r0_mask & xmask, mean_next, mean_acc)
        m2_acc = tl.where(r0_mask & xmask, m2_next, m2_acc)
        weight_acc = tl.where(r0_mask & xmask, weight_next, weight_acc)

    block_mean, block_m2, block_weight = triton_helpers.welford(
        mean_acc,
        m2_acc,
        weight_acc,
        1,
    )
    out_offsets = channel + 768 * hw
    plane_stride = 37632
    tl.store(partial_ptr + out_offsets, block_mean[:, None], xmask)
    tl.store(partial_ptr + plane_stride + out_offsets, block_m2[:, None], xmask)
    tl.store(partial_ptr + 2 * plane_stride + out_offsets, block_weight[:, None], xmask)


@triton_heuristics.persistent_reduction(
    size_hints={"x": 1024, "r0_": 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta=_triton_meta(
        {
            "partial_ptr": "*fp32",
            "running_mean_ptr": "*fp32",
            "running_var_ptr": "*fp32",
            "mean_ptr": "*fp32",
            "invstd_ptr": "*fp32",
            "xnumel": "i32",
            "r0_numel": "i32",
            "XBLOCK": "constexpr",
        }
    ),
    inductor_meta={
        "grid_type": "Grid1D",
        "mutated_arg_names": ["running_mean_ptr", "running_var_ptr"],
        "optimize_mem": True,
        "autotune_pointwise": True,
        "coordinate_descent_tuning": True,
    },
)
@triton.jit
def _d9_finalize_kernel(
    partial_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
):
    xnumel = 768
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    offsets = xindex + 768 * r0_index
    plane_stride = 37632
    mean_inputs = tl.load(partial_ptr + offsets, r0_mask & xmask, eviction_policy="evict_first", other=0.0)
    m2_inputs = tl.load(
        partial_ptr + plane_stride + offsets,
        r0_mask & xmask,
        eviction_policy="evict_first",
        other=0.0,
    )
    weight_inputs = tl.load(
        partial_ptr + 2 * plane_stride + offsets,
        r0_mask & xmask,
        eviction_policy="evict_first",
        other=0.0,
    )
    mean_inputs = tl.where(r0_mask & xmask, mean_inputs, 0.0)
    m2_inputs = tl.where(r0_mask & xmask, m2_inputs, 0.0)
    weight_inputs = tl.where(r0_mask & xmask, weight_inputs, 0.0)
    mean, m2, _weight = triton_helpers.welford(mean_inputs, m2_inputs, weight_inputs, 1)
    mean = mean[:, None]
    var = (m2[:, None] / 6272.0)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_var = tl.load(running_var_ptr + xindex, xmask)
    old_mean = tl.load(running_mean_ptr + xindex, xmask)
    new_var = _f32_add(_f32_mul(_f32_mul(var, 1.0001594642002871), 0.1), _f32_mul(old_var, 0.9))
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))

    tl.store(mean_ptr + xindex, mean, xmask)
    tl.store(invstd_ptr + xindex, invstd, xmask)
    tl.store(running_var_ptr + xindex, new_var, xmask)
    tl.store(running_mean_ptr + xindex, new_mean, xmask)


@triton_heuristics.pointwise(
    size_hints={"x": 8388608},
    filename=__file__,
    triton_meta=_triton_meta(
        {
            "x0_ptr": "*fp32",
            "x1_ptr": "*bf16",
            "mean_ptr": "*fp32",
            "invstd_ptr": "*fp32",
            "weight_ptr": "*fp32",
            "bias_ptr": "*fp32",
            "add_out_ptr": "*fp32",
            "affine_out_ptr": "*bf16",
            "centered_out_ptr": "*fp32",
            "xnumel": "i32",
            "XBLOCK": "constexpr",
        }
    ),
    inductor_meta={
        "grid_type": "Grid1D",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "autotune_pointwise": True,
        "coordinate_descent_tuning": True,
    },
    min_elem_per_thread=0,
)
@triton.jit
def _d9_recompute_epilogue_kernel(
    x0_ptr,
    x1_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    affine_out_ptr,
    centered_out_ptr,
    xnumel,
    XBLOCK: tl.constexpr,
):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    channel = xindex % 768

    x0 = tl.load(x0_ptr + xindex, xmask).to(tl.float32)
    x1 = tl.load(x1_ptr + xindex, xmask).to(tl.float32)
    mean = tl.load(mean_ptr + channel, xmask, eviction_policy="evict_last")
    invstd = tl.load(invstd_ptr + channel, xmask, eviction_policy="evict_last")
    weight = tl.load(weight_ptr + channel, xmask, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + channel, xmask, eviction_policy="evict_last").to(tl.float32)

    added = _f32_add(x0, x1)
    centered = _f32_sub(added, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    tl.store(add_out_ptr + xindex, added, xmask)
    tl.store(centered_out_ptr + xindex, centered, xmask)
    tl.store(
        affine_out_ptr + xindex,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        xmask,
    )

@triton.jit
def _partial_add_stats_kernel(
    x0_ptr,
    x1_ptr,
    add_out_ptr,
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
    n = r_offsets // hw_size
    hw = r_offsets - n * hw_size
    h = hw // width
    w = hw - h * width

    x0_offsets = (
        n[None, :] * x0_stride_n
        + c_offsets[:, None] * x0_stride_c
        + h[None, :] * x0_stride_h
        + w[None, :] * x0_stride_w
    )
    x1_offsets = (
        n[None, :] * x1_stride_n
        + c_offsets[:, None] * x1_stride_c
        + h[None, :] * x1_stride_h
        + w[None, :] * x1_stride_w
    )
    mask = c_mask[:, None] & r_mask[None, :]

    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)
    added = _f32_add(x0, x1)
    tl.store(add_out_ptr + x0_offsets, added, mask=mask)

    sums = tl.sum(added, axis=1)
    sums2 = tl.sum(_f32_mul(added, added), axis=1)
    out_offsets = chunk * channels + c_offsets
    plane_stride = num_chunks * channels
    tl.store(partial_ptr + out_offsets, sums, mask=c_mask)
    tl.store(partial_ptr + plane_stride + out_offsets, sums2, mask=c_mask)


@triton.jit
def _finalize_stats_kernel(
    partial_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (c_offsets[:, None] < channels) & (chunks[None, :] < num_chunks)
    offsets = chunks[None, :] * channels + c_offsets[:, None]
    plane_stride = num_chunks * channels

    sums = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0), axis=1)
    sums2 = tl.sum(
        tl.load(partial_ptr + plane_stride + offsets, mask=mask, other=0.0),
        axis=1,
    )
    inv_count = 1.0 / elements_per_channel
    mean = _f32_mul(sums, inv_count)
    var = _f32_sub(_f32_mul(sums2, inv_count), _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    channel_mask = c_offsets < channels
    old_mean = tl.load(
        running_mean_ptr + c_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    old_var = tl.load(
        running_var_ptr + c_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(var, 1.0001594642002871)
    new_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))

    tl.store(mean_ptr + c_offsets, mean, mask=channel_mask)
    tl.store(invstd_ptr + c_offsets, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + c_offsets, new_mean, mask=channel_mask)
    tl.store(running_var_ptr + c_offsets, new_var, mask=channel_mask)


@triton.jit
def _affine_centered_kernel(
    add_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    affine_out_ptr,
    centered_out_ptr,
    total_elements: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    is_channels_last: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < total_elements

    if is_channels_last:
        channel = offsets - (offsets // channels) * channels
    else:
        channel = (offsets // hw_size) - ((offsets // hw_size) // channels) * channels

    added = tl.load(add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    centered = _f32_sub(added, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    tl.store(centered_out_ptr + offsets, centered, mask=mask)
    tl.store(
        affine_out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="d9d8b8eb", BLOCK_R=2048, BLOCK_C=16, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_epi=4, USE_D9_HEURISTIC=True)
@oracle_impl(hardware="B200", point="cf9ba6ac", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_epi=4)
@oracle_impl(hardware="B200", point="e3b6f6ad", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_epi=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_R,
    BLOCK_C,
    BLOCK_E,
    num_warps_stats,
    num_warps_final,
    num_warps_epi,
    USE_D9_HEURISTIC=False,
):
    x0, x1, running_mean, running_var, weight, bias = inputs
    batch = int(x0.shape[0])
    channels = int(x0.shape[1])
    height = int(x0.shape[2])
    width = int(x0.shape[3])
    hw_size = height * width
    elements_per_channel = batch * hw_size
    total_elements = batch * channels * hw_size
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    add_out = torch.empty_strided(
        tuple(x0.shape),
        tuple(x0.stride()),
        device=x0.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (channels,),
        (1,),
        device=x0.device,
        dtype=torch.float32,
    )
    affine_out = torch.empty_strided(
        tuple(x0.shape),
        tuple(x0.stride()),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    centered_out = torch.empty_strided(
        tuple(x0.shape),
        tuple(x0.stride()),
        device=x0.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (channels,),
        (1,),
        device=x0.device,
        dtype=torch.float32,
    )

    if USE_D9_HEURISTIC:
        partial = torch.empty_strided(
            (3, channels, hw_size),
            (channels * hw_size, 1, channels),
            device=x0.device,
            dtype=torch.float32,
        )
        stream = get_raw_stream(x0.get_device())
        _d9_partial_add_welford_kernel.run(
            x0,
            x1,
            partial,
            channels * hw_size,
            batch,
            stream=stream,
        )
        _d9_finalize_kernel.run(
            partial,
            running_mean,
            running_var,
            mean,
            invstd,
            channels,
            hw_size,
            stream=stream,
        )
        _d9_recompute_epilogue_kernel.run(
            x0,
            x1,
            mean,
            invstd,
            weight,
            bias,
            add_out,
            affine_out,
            centered_out,
            total_elements,
            stream=stream,
        )
        return add_out, invstd, affine_out, centered_out, running_mean, running_var

    partial = torch.empty_strided(
        (2, num_chunks, channels),
        (num_chunks * channels, channels, 1),
        device=x0.device,
        dtype=torch.float32,
    )

    _partial_add_stats_kernel[(triton.cdiv(channels, BLOCK_C), num_chunks)](
        x0,
        x1,
        add_out,
        partial,
        channels=channels,
        width=width,
        hw_size=hw_size,
        elements_per_channel=elements_per_channel,
        x0_stride_n=int(x0.stride(0)),
        x0_stride_c=int(x0.stride(1)),
        x0_stride_h=int(x0.stride(2)),
        x0_stride_w=int(x0.stride(3)),
        x1_stride_n=int(x1.stride(0)),
        x1_stride_c=int(x1.stride(1)),
        x1_stride_h=int(x1.stride(2)),
        x1_stride_w=int(x1.stride(3)),
        num_chunks=num_chunks,
        BLOCK_C=BLOCK_C,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _finalize_stats_kernel[(triton.cdiv(channels, BLOCK_C),)](
        partial,
        running_mean,
        running_var,
        mean,
        invstd,
        channels=channels,
        elements_per_channel=elements_per_channel,
        num_chunks=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_final,
        num_stages=3,
    )
    _affine_centered_kernel[(triton.cdiv(total_elements, BLOCK_E),)](
        add_out,
        mean,
        invstd,
        weight,
        bias,
        affine_out,
        centered_out,
        total_elements=total_elements,
        channels=channels,
        hw_size=hw_size,
        is_channels_last=int(x0.stride(1)) == 1,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps_epi,
        num_stages=3,
    )
    return add_out, invstd, affine_out, centered_out, running_mean, running_var
