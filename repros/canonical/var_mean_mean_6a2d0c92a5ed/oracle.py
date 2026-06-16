"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 training-BatchNorm hard-swish spatial-mean scope, including bf16-to-fp32 population var_mean over N,H,W, eps=1e-5 rsqrt, saved mean and rsqrt side outputs, mutable running-stat copy_ aliases with the captured variance-correction literal, explicit bf16 affine rounding before hard-swish, the returned channels-last bf16 activation, and the returned bf16 spatial mean reduced from that activation, whereas Inductor lowers the statistics, running-stat updates, visible activation materialization, and downstream spatial reduction through generic separated schedules; Inductor cannot do this today because its normalization scheduler does not keep multi-output BN-training statistics, side-effecting running-stat updates, visible bf16 hard-swish activation, and a sibling spatial-reduction consumer in one channels-last plan; the fix is SCHEDULER_FUSION: extend the BN-training scheduler to emit saved stats and mutable running-stat epilogues while sinking fixed-shape bf16 hard-swish and spatial-mean consumers into the same planned lowering."""

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
def _partial_stats_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sum2_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    r_block = tl.program_id(1)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
    hw_size: tl.constexpr = H * W
    hw = r_offsets - (r_offsets // hw_size) * hw_size
    n_idx = r_offsets // hw_size
    h_idx = hw // W
    w_idx = hw - h_idx * W
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    mask = (channels[:, None] < C) & (r_offsets[None, :] < E)
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    sums = tl.sum(values, axis=1)
    sums2 = tl.sum(_f32_mul(values, values), axis=1)
    out_offsets = r_block * C + channels
    tl.store(partial_sum_ptr + out_offsets, sums, mask=channels < C)
    tl.store(partial_sum2_ptr + out_offsets, sums2, mask=channels < C)


@triton.jit
def _finalize_stats_kernel(
    partial_sum_ptr,
    partial_sum2_ptr,
    running_mean_ptr,
    running_var_ptr,
    saved_mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    channel_mask = channels < C
    mask = channel_mask[:, None] & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums2 = tl.load(partial_sum2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(sums, axis=1)
    total2 = tl.sum(sums2, axis=1)

    mean = total / E
    var = _f32_sub(total2 / E, _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(var, 1.0000398612827361)
    new_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))

    tl.store(saved_mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=channel_mask)
    tl.store(running_var_ptr + channels, new_var, mask=channel_mask)


@triton.jit
def _hardswish_spatial_mean_kernel(
    x_ptr,
    saved_mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    activation_ptr,
    spatial_mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    hw_offsets = tl.arange(0, BLOCK_HW)
    channels = rows - (rows // C) * C
    n_idx = rows // C
    h_idx = hw_offsets // W
    w_idx = hw_offsets - h_idx * W
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw_offsets < (H * W)
    mask = row_mask[:, None] & hw_mask[None, :]
    x_offsets = (
        n_idx[:, None] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    out_offsets = (
        n_idx[:, None] * out_stride_n
        + channels[:, None] * out_stride_c
        + h_idx[None, :] * out_stride_h
        + w_idx[None, :] * out_stride_w
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(saved_mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    normalized = _f32_mul(_f32_sub(x, mean[:, None]), invstd[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[:, None]), bias[:, None])
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    relu6 = tl.minimum(tl.maximum(_f32_add(rounded, 3.0), 0.0), 6.0)
    hardswish = _f32_mul(_f32_mul(rounded, relu6), 0.16666666666666666)
    hardswish_bf16 = hardswish.to(tl.bfloat16)

    tl.store(activation_ptr + out_offsets, hardswish_bf16, mask=mask)
    reduced = tl.sum(tl.where(mask, hardswish_bf16.to(tl.float32), 0.0), axis=1)
    mean_value = (reduced / (H * W)).to(tl.bfloat16)
    tl.store(spatial_mean_ptr + rows, mean_value, mask=row_mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# d6a317bc: (T([512,960,7,7], bf16, stride=(47040,1,6720,960)), T([960], f32), T([960], f32), T([960], f32), T([960], f32))
@oracle_impl(hardware="B200", point="d6a317bc", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=64, ROW_BLOCK=32, STAT_WARPS=8, FINAL_WARPS=1, OUT_WARPS=2)
# c5a5573b: (T([512,672,7,7], bf16, stride=(32928,1,4704,672)), T([672], f32), T([672], f32), T([672], f32), T([672], f32))
@oracle_impl(hardware="B200", point="c5a5573b", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=64, ROW_BLOCK=32, STAT_WARPS=8, FINAL_WARPS=1, OUT_WARPS=2)
# 055f7aad: (T([512,672,14,14], bf16, stride=(131712,1,9408,672)), T([672], f32), T([672], f32), T([672], f32), T([672], f32))
@oracle_impl(hardware="B200", point="055f7aad", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=256, ROW_BLOCK=8, STAT_WARPS=8, FINAL_WARPS=1, OUT_WARPS=8)
# bb4cfa64: (T([512,480,14,14], bf16, stride=(94080,1,6720,480)), T([480], f32), T([480], f32), T([480], f32), T([480], f32))
@oracle_impl(hardware="B200", point="bb4cfa64", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=256, ROW_BLOCK=8, STAT_WARPS=8, FINAL_WARPS=1, OUT_WARPS=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    ROW_BLOCK: int,
    STAT_WARPS: int,
    FINAL_WARPS: int,
    OUT_WARPS: int,
):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    elements_per_channel = n * h * w
    total_rows = n * c
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    saved_mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    activation = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    spatial_mean = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.bfloat16)
    partial_sum = torch.empty_strided((num_chunks, c), (c, 1), device=x.device, dtype=torch.float32)
    partial_sum2 = torch.empty_strided((num_chunks, c), (c, 1), device=x.device, dtype=torch.float32)

    _partial_stats_kernel[(triton.cdiv(c, BLOCK_C), num_chunks)](
        x,
        partial_sum,
        partial_sum2,
        C=c,
        H=h,
        W=w,
        E=elements_per_channel,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=STAT_WARPS,
        num_stages=3,
    )
    _finalize_stats_kernel[(triton.cdiv(c, BLOCK_C),)](
        partial_sum,
        partial_sum2,
        running_mean,
        running_var,
        saved_mean,
        invstd,
        C=c,
        E=elements_per_channel,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=BLOCK_C,
        num_warps=FINAL_WARPS,
        num_stages=3,
    )
    _hardswish_spatial_mean_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        x,
        saved_mean,
        invstd,
        weight,
        bias,
        activation,
        spatial_mean,
        C=c,
        H=h,
        W=w,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        out_stride_n=int(activation.stride(0)),
        out_stride_c=int(activation.stride(1)),
        out_stride_h=int(activation.stride(2)),
        out_stride_w=int(activation.stride(3)),
        TOTAL_ROWS=total_rows,
        BLOCK_HW=BLOCK_HW,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=3,
    )
    return saved_mean, invstd, activation, spatial_mean, running_mean, running_var
