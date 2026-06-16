"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet bf16 training-BatchNorm natural-exp SiLU plus spatial-mean scope, including bf16-to-fp32 population var_mean over N,H,W, eps=0.001 rsqrt, saved mean and rsqrt side outputs, mutable running-stat copy_ aliases with the captured variance-correction literal, Inductor's f32-affine natural-exp SiLU placement, returned bf16 activation, and the returned bf16 spatial mean reduced from that activation, whereas Inductor lowers the statistics, running-stat updates, activation materialization, and downstream spatial reduction through generic separated schedules; Inductor cannot do this today because its normalization scheduler does not keep multi-output BN-training statistics, side-effecting running-stat updates, a visible bf16 activation producer, and a sibling spatial-reduction consumer in one channels-last plan; the fix is SCHEDULER_FUSION: extend the BN-training scheduler to emit saved stats and mutable running-stat epilogues while sinking fixed-shape bf16 SiLU and spatial-mean consumers into the same planned lowering."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

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
def _direct_stats_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = tl.arange(0, BLOCK_R)
    hw_size: tl.constexpr = H * W
    hw = r_offsets - (r_offsets // hw_size) * hw_size
    n_idx = r_offsets // hw_size
    h_idx = hw // W
    w_idx = hw - h_idx * W
    c_mask = channels < C
    r_mask = r_offsets < E
    mask = c_mask[:, None] & r_mask[None, :]
    x_offsets = (
        n_idx[None, :] * x_s0
        + channels[:, None] * x_s1
        + h_idx[None, :] * x_s2
        + w_idx[None, :] * x_s3
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(x, axis=1)
    sum_x2 = tl.sum(_f32_mul(x, x), axis=1)
    inv_e: tl.constexpr = 1.0 / E
    mean = _f32_mul(sum_x, inv_e)
    var = _f32_sub(_f32_mul(sum_x2, inv_e), _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 0.001))

    old_mean = tl.load(running_mean_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    new_var = _f32_add(
        _f32_mul(_f32_mul(var, 1.0001594642002871), 0.1),
        _f32_mul(old_var, 0.9),
    )

    tl.store(mean_ptr + channels, mean, mask=c_mask)
    tl.store(invstd_ptr + channels, invstd, mask=c_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=c_mask)
    tl.store(running_var_ptr + channels, new_var, mask=c_mask)


@triton.jit
def _partial_sums_kernel(
    x_ptr,
    partial_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    hw_size: tl.constexpr = H * W
    hw = r_offsets - (r_offsets // hw_size) * hw_size
    n_idx = r_offsets // hw_size
    h_idx = hw // W
    w_idx = hw - h_idx * W
    c_mask = channels < C
    r_mask = r_offsets < E
    mask = c_mask[:, None] & r_mask[None, :]
    x_offsets = (
        n_idx[None, :] * x_s0
        + channels[:, None] * x_s1
        + h_idx[None, :] * x_s2
        + w_idx[None, :] * x_s3
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    sum_x = tl.sum(x, axis=1)
    sum_x2 = tl.sum(_f32_mul(x, x), axis=1)
    offsets = tl.program_id(1) * C + channels
    plane = NUM_CHUNKS * C
    tl.store(partial_ptr + offsets, sum_x, mask=c_mask)
    tl.store(partial_ptr + plane + offsets, sum_x2, mask=c_mask)


@triton.jit
def _finalize_stats_kernel(
    partial_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    c_mask = channels < C
    mask = c_mask[:, None] & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    plane = NUM_CHUNKS * C
    sum_x = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=1)
    sum_x2 = tl.sum(tl.load(partial_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=1)
    inv_e: tl.constexpr = 1.0 / E
    mean = _f32_mul(sum_x, inv_e)
    var = _f32_sub(_f32_mul(sum_x2, inv_e), _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 0.001))

    old_mean = tl.load(running_mean_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    new_var = _f32_add(
        _f32_mul(_f32_mul(var, 1.0001594642002871), 0.1),
        _f32_mul(old_var, 0.9),
    )

    tl.store(mean_ptr + channels, mean, mask=c_mask)
    tl.store(invstd_ptr + channels, invstd, mask=c_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=c_mask)
    tl.store(running_var_ptr + channels, new_var, mask=c_mask)


@triton.jit
def _silu_spatial_mean_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    spatial_mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
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
    offsets = (
        n_idx[:, None] * x_s0
        + channels[:, None] * x_s1
        + h_idx[None, :] * x_s2
        + w_idx[None, :] * x_s3
    )

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    norm = (x - mean[:, None]) * invstd[:, None]
    affine = norm * weight[:, None] + bias[:, None]
    denom = tl_math.exp(-affine) + 1.0
    silu = affine / denom
    silu_bf16 = silu.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + offsets, silu_bf16, mask=mask)
    reduced = tl.sum(tl.where(mask, silu_bf16.to(tl.float32), 0.0), axis=1)
    tl.store(spatial_mean_ptr + rows, (reduced / (H * W)).to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=row_mask)


@triton.jit
def _silu_flat_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    channels = offsets - (offsets // C) * C
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)

    norm = (x - mean) * invstd
    affine = norm * weight + bias
    denom = tl_math.exp(-affine) + 1.0
    silu = affine / denom
    tl.store(out_ptr + offsets, silu.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@triton.jit
def _spatial_mean_from_activation_kernel(
    out_ptr,
    spatial_mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    out_s3: tl.constexpr,
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
    offsets = (
        n_idx[:, None] * out_s0
        + channels[:, None] * out_s1
        + h_idx[None, :] * out_s2
        + w_idx[None, :] * out_s3
    )
    values = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    reduced = tl.sum(values, axis=1)
    tl.store(spatial_mean_ptr + rows, (reduced / (H * W)).to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=row_mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="c2791544", DIRECT=1, DIRECT_BLOCK_C=8, STATS_BLOCK_R=2048, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=0, BLOCK_HW=64, ROW_BLOCK=32, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4, mean_warps=2)
@oracle_impl(hardware="B200", point="dc4c8729", DIRECT=1, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=1, BLOCK_HW=64, ROW_BLOCK=16, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4, mean_warps=2)
@oracle_impl(hardware="B200", point="e41460a6", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=1, BLOCK_HW=256, ROW_BLOCK=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4, mean_warps=4)
@oracle_impl(hardware="B200", point="886f27c9", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=1, BLOCK_HW=256, ROW_BLOCK=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4, mean_warps=4)
@oracle_impl(hardware="B200", point="8ffa8a4f", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=1, BLOCK_HW=256, ROW_BLOCK=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4, mean_warps=4)
@oracle_impl(hardware="B200", point="cdca2f80", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=2048, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=1, BLOCK_HW=1024, ROW_BLOCK=4, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=8, mean_warps=8)
@oracle_impl(hardware="B200", point="9d6ef1f8", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=2048, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=1, BLOCK_HW=1024, ROW_BLOCK=4, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=8, mean_warps=8)
@oracle_impl(hardware="B200", point="46238279", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=0, BLOCK_HW=4096, ROW_BLOCK=2, ACT_BLOCK=1024, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4, mean_warps=8)
@oracle_impl(hardware="B200", point="21476974", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=0, BLOCK_HW=4096, ROW_BLOCK=2, ACT_BLOCK=1024, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4, mean_warps=8)
@oracle_impl(hardware="B200", point="bf1cc8fb", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, FUSED_MEAN=0, BLOCK_HW=16384, ROW_BLOCK=1, ACT_BLOCK=1024, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4, mean_warps=8)
def oracle_forward(
    inputs,
    *,
    DIRECT: int,
    DIRECT_BLOCK_C: int,
    STATS_BLOCK_R: int,
    STATS_BLOCK_C: int,
    FINAL_BLOCK_C: int,
    FUSED_MEAN: int,
    BLOCK_HW: int,
    ROW_BLOCK: int,
    ACT_BLOCK: int,
    direct_warps: int,
    stats_warps: int,
    final_warps: int,
    act_warps: int,
    mean_warps: int,
):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    e = n * hw
    total_rows = n * c
    total = total_rows * hw

    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    spatial_mean = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.bfloat16)

    if DIRECT:
        _direct_stats_kernel[(triton.cdiv(c, DIRECT_BLOCK_C),)](
            x,
            running_mean,
            running_var,
            mean,
            invstd,
            C=c,
            H=h,
            W=w,
            E=e,
            x_s0=x.stride(0),
            x_s1=x.stride(1),
            x_s2=x.stride(2),
            x_s3=x.stride(3),
            BLOCK_C=DIRECT_BLOCK_C,
            BLOCK_R=_next_power_of_2(e),
            num_warps=direct_warps,
            num_stages=3,
        )
    else:
        num_chunks = triton.cdiv(e, STATS_BLOCK_R)
        partial = torch.empty((2, num_chunks, c), device=x.device, dtype=torch.float32)
        _partial_sums_kernel[(triton.cdiv(c, STATS_BLOCK_C), num_chunks)](
            x,
            partial,
            C=c,
            H=h,
            W=w,
            E=e,
            x_s0=x.stride(0),
            x_s1=x.stride(1),
            x_s2=x.stride(2),
            x_s3=x.stride(3),
            NUM_CHUNKS=num_chunks,
            BLOCK_C=STATS_BLOCK_C,
            BLOCK_R=STATS_BLOCK_R,
            num_warps=stats_warps,
            num_stages=3,
        )
        _finalize_stats_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
            partial,
            running_mean,
            running_var,
            mean,
            invstd,
            C=c,
            E=e,
            NUM_CHUNKS=num_chunks,
            BLOCK_CHUNKS=_next_power_of_2(num_chunks),
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=final_warps,
            num_stages=3,
        )

    if FUSED_MEAN:
        _silu_spatial_mean_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
            x,
            weight,
            bias,
            mean,
            invstd,
            out,
            spatial_mean,
            C=c,
            H=h,
            W=w,
            x_s0=x.stride(0),
            x_s1=x.stride(1),
            x_s2=x.stride(2),
            x_s3=x.stride(3),
            TOTAL_ROWS=total_rows,
            BLOCK_HW=BLOCK_HW,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=act_warps,
            num_stages=3,
        )
    else:
        _silu_flat_kernel[(triton.cdiv(total, ACT_BLOCK),)](
            x,
            weight,
            bias,
            mean,
            invstd,
            out,
            TOTAL=total,
            C=c,
            BLOCK=ACT_BLOCK,
            num_warps=act_warps,
            num_stages=3,
        )
        _spatial_mean_from_activation_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
            out,
            spatial_mean,
            C=c,
            H=h,
            W=w,
            out_s0=out.stride(0),
            out_s1=out.stride(1),
            out_s2=out.stride(2),
            out_s3=out.stride(3),
            TOTAL_ROWS=total_rows,
            BLOCK_HW=BLOCK_HW,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=mean_warps,
            num_stages=3,
        )

    return mean, invstd, out, spatial_mean, running_mean, running_var
