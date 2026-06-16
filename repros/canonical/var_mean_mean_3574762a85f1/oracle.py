"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 GhostNet training-BatchNorm plus ReLU spatial-mean scope, including bf16-to-fp32 per-channel population statistics over N/H/W, eps=1e-5 rsqrt and saved mean side outputs, mutable running-stat copy_ updates with the captured correction literal, fp32 affine math, bf16 rounding before ReLU, final bf16 7x7 spatial mean with the captured as_strided layout, and returned running-stat aliases, whereas Inductor lowers the norm-template statistics/update region and downstream affine/ReLU/mean consumer through generic normalization and reduction schedules; Inductor cannot do this today because the scheduler does not keep training-BN side outputs, mutable stat epilogues, the required bf16 cast boundary, and the spatial-mean consumer in one full-scope channel plan; the fix is SCHEDULER_FUSION: extend the BatchNorm training template to emit saved statistics and running-stat aliases while fusing affine, bf16 ReLU, and fixed-spatial mean stores from the same schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _bn_partial_stats_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sq_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_offsets = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    hw: tl.constexpr = H * W
    n_idx = e_offsets // hw
    spatial = e_offsets - n_idx * hw
    h_idx = spatial // W
    w_idx = spatial - h_idx * W
    mask = (c_offsets[:, None] < C) & (e_offsets[None, :] < E)
    offsets = (
        n_idx[None, :] * SN
        + c_offsets[:, None] * SC
        + h_idx[None, :] * SH
        + w_idx[None, :] * SW
    )
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    active = tl.where(mask, vals, 0.0)
    sums = tl.sum(active, axis=1)
    sums_sq = tl.sum(_f32_mul(active, active), axis=1)
    out_offsets = tl.program_id(1) * C + c_offsets
    tl.store(partial_sum_ptr + out_offsets, sums, mask=c_offsets < C)
    tl.store(partial_sq_ptr + out_offsets, sums_sq, mask=c_offsets < C)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_sum_ptr,
    partial_sq_ptr,
    running_mean_ptr,
    running_var_ptr,
    saved_mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_offsets = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (c_offsets[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + c_offsets[:, None]
    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums_sq = tl.load(partial_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(mask, sums, 0.0), axis=1)
    total_sq = tl.sum(tl.where(mask, sums_sq, 0.0), axis=1)
    mean = total / E
    ex2 = total_sq / E
    var = tl.maximum(_f32_sub(ex2, _f32_mul(mean, mean)), 0.0)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    c_mask = c_offsets < C
    old_mean = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    new_var = _f32_add(
        _f32_mul(_f32_mul(var, 1.0000398612827361), 0.1),
        _f32_mul(old_var, 0.9),
    )
    tl.store(saved_mean_ptr + c_offsets, mean, mask=c_mask)
    tl.store(invstd_ptr + c_offsets, invstd, mask=c_mask)
    tl.store(running_mean_ptr + c_offsets, new_mean, mask=c_mask)
    tl.store(running_var_ptr + c_offsets, new_var, mask=c_mask)


@triton.jit
def _bn_relu_spatial_mean_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    saved_mean_ptr,
    invstd_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL_ROWS_N: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw_offsets = tl.arange(0, BLOCK_HW)
    hw: tl.constexpr = H * W
    n_idx = rows // C
    channels = rows - n_idx * C
    h_idx = hw_offsets // W
    w_idx = hw_offsets - h_idx * W
    row_mask = rows < TOTAL_ROWS_N
    hw_mask = hw_offsets < hw
    mask = row_mask[:, None] & hw_mask[None, :]
    offsets = (
        n_idx[:, None] * SN
        + channels[:, None] * SC
        + h_idx[None, :] * SH
        + w_idx[None, :] * SW
    )

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(saved_mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    scaled = _f32_mul(normalized, weight[:, None])
    affine = _f32_add(scaled, bias[:, None]).to(tl.bfloat16)
    zero = tl.full((BLOCK_ROWS, BLOCK_HW), 0.0, tl.bfloat16)
    relu = tl.where(affine < zero, zero, affine)
    pooled = tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1) * (1.0 / hw)
    tl.store(out_ptr + rows, pooled.to(tl.bfloat16), mask=row_mask)


# d6a317bc: GhostNet training BN, bf16 channels-last input [512,960,7,7].
@oracle_impl(hardware="B200", point="d6a317bc", BLOCK_E=4096, C_BLOCK=32, OUT_BLOCK_ROWS=64, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    C_BLOCK: int,
    OUT_BLOCK_ROWS: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1 = inputs
    n = arg0_1.shape[0]
    c = arg0_1.shape[1]
    h = arg0_1.shape[2]
    w = arg0_1.shape[3]
    e = n * h * w
    num_chunks = triton.cdiv(e, BLOCK_E)
    block_chunks = _next_power_of_2(num_chunks)

    saved_mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    spatial_mean = torch.empty_strided((n, c, 1, 1), (c, 1, c, c), device=arg0_1.device, dtype=torch.bfloat16)
    partial_sum = torch.empty((num_chunks, c), device=arg0_1.device, dtype=torch.float32)
    partial_sq = torch.empty((num_chunks, c), device=arg0_1.device, dtype=torch.float32)

    _bn_partial_stats_kernel[(triton.cdiv(c, C_BLOCK), num_chunks)](
        arg0_1,
        partial_sum,
        partial_sq,
        c,
        h,
        w,
        e,
        arg0_1.stride(0),
        arg0_1.stride(1),
        arg0_1.stride(2),
        arg0_1.stride(3),
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _bn_finalize_stats_kernel[(triton.cdiv(c, C_BLOCK),)](
        partial_sum,
        partial_sq,
        arg1_1,
        arg2_1,
        saved_mean,
        invstd,
        c,
        e,
        num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=C_BLOCK,
        num_warps=1,
        num_stages=3,
    )
    _bn_relu_spatial_mean_kernel[(triton.cdiv(n * c, OUT_BLOCK_ROWS),)](
        arg0_1,
        arg3_1,
        arg4_1,
        saved_mean,
        invstd,
        spatial_mean,
        c,
        h,
        w,
        n * c,
        arg0_1.stride(0),
        arg0_1.stride(1),
        arg0_1.stride(2),
        arg0_1.stride(3),
        BLOCK_ROWS=OUT_BLOCK_ROWS,
        BLOCK_HW=triton.next_power_of_2(h * w),
        num_warps=2,
        num_stages=3,
    )
    return saved_mean, invstd, spatial_mean, arg1_1, arg2_1
