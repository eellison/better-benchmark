"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 dual training-BatchNorm block, including both bf16-to-fp32 per-channel population var_mean reductions, eps=1e-5 rsqrt side outputs, running-stat copy_ side effects with the captured correction literal, first-BN bf16 affine rounding through the channel-cat half, second-BN bf16 affine rounding, final bf16 residual add, and the returned mean view tensors, whereas Inductor schedules the two norm-template reductions and the downstream channel-cat/add epilogue as separate generic regions; Inductor cannot do this today because the scheduler does not canonicalize sibling BN-training reductions feeding a layout-sensitive channel cat plus residual add into one multi-output channels-last norm epilogue; the fix is SCHEDULER_FUSION: add a guarded dual-BN training template that batches stats finalization and fuses the cat/add output writer while preserving running-stat updates, bf16 cast boundaries, and side-output views."""

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
def _dual_bn_partial_stats_kernel(
    x1_ptr,
    x2_ptr,
    partial_sum_ptr,
    partial_sum2_ptr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride1_n: tl.constexpr,
    stride1_c: tl.constexpr,
    stride1_h: tl.constexpr,
    stride1_w: tl.constexpr,
    stride2_n: tl.constexpr,
    stride2_c: tl.constexpr,
    stride2_h: tl.constexpr,
    stride2_w: tl.constexpr,
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_offsets = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    total_c = C1 + C2
    first_bn = c_offsets < C1
    channels1 = c_offsets
    channels2 = c_offsets - C1

    hw = e_offsets % (H * W)
    n_idx = e_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    mask = (c_offsets[:, None] < total_c) & (e_offsets[None, :] < E)

    offsets1 = (
        n_idx[None, :] * stride1_n
        + channels1[:, None] * stride1_c
        + h_idx[None, :] * stride1_h
        + w_idx[None, :] * stride1_w
    )
    offsets2 = (
        n_idx[None, :] * stride2_n
        + channels2[:, None] * stride2_c
        + h_idx[None, :] * stride2_h
        + w_idx[None, :] * stride2_w
    )
    vals1 = tl.load(x1_ptr + offsets1, mask=mask & first_bn[:, None], other=0.0).to(tl.float32)
    vals2 = tl.load(x2_ptr + offsets2, mask=mask & ~first_bn[:, None], other=0.0).to(tl.float32)
    vals = vals1 + vals2

    sums = tl.sum(vals, axis=1)
    sums2 = tl.sum(vals * vals, axis=1)
    out_offsets = tl.program_id(1) * total_c + c_offsets
    tl.store(partial_sum_ptr + out_offsets, sums, mask=c_offsets < total_c)
    tl.store(partial_sum2_ptr + out_offsets, sums2, mask=c_offsets < total_c)


@triton.jit
def _dual_bn_finalize_stats_kernel(
    partial_sum_ptr,
    partial_sum2_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    running_mean2_ptr,
    running_var2_ptr,
    mean1_ptr,
    invstd1_ptr,
    mean2_ptr,
    invstd2_ptr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_offsets = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    total_c = C1 + C2
    first_bn = c_offsets < C1
    channels1 = c_offsets
    channels2 = c_offsets - C1
    mask = (c_offsets[:, None] < total_c) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * total_c + c_offsets[:, None]
    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)
    sums2 = tl.load(partial_sum2_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(sums, axis=1).to(tl.float32)
    total2 = tl.sum(sums2, axis=1).to(tl.float32)
    mean = total / E
    var = total2 / E - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = libdevice.rsqrt(var + 1.0e-5)
    channel_mask = c_offsets < total_c

    old_mean1 = tl.load(running_mean1_ptr + channels1, mask=channel_mask & first_bn, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + channels1, mask=channel_mask & first_bn, other=0.0).to(tl.float32)
    old_mean2 = tl.load(running_mean2_ptr + channels2, mask=channel_mask & ~first_bn, other=0.0).to(tl.float32)
    old_var2 = tl.load(running_var2_ptr + channels2, mask=channel_mask & ~first_bn, other=0.0).to(tl.float32)

    mean_update = _f32_mul(mean, 0.1)
    var_update = _f32_mul(_f32_mul(var, 1.0000398612827361), 0.1)
    new_mean1 = _f32_add(_f32_mul(old_mean1, 0.9), mean_update)
    new_var1 = _f32_add(_f32_mul(old_var1, 0.9), var_update)
    new_mean2 = _f32_add(_f32_mul(old_mean2, 0.9), mean_update)
    new_var2 = _f32_add(_f32_mul(old_var2, 0.9), var_update)

    tl.store(running_mean1_ptr + channels1, new_mean1, mask=channel_mask & first_bn)
    tl.store(running_var1_ptr + channels1, new_var1, mask=channel_mask & first_bn)
    tl.store(mean1_ptr + channels1, mean, mask=channel_mask & first_bn)
    tl.store(invstd1_ptr + channels1, invstd, mask=channel_mask & first_bn)
    tl.store(running_mean2_ptr + channels2, new_mean2, mask=channel_mask & ~first_bn)
    tl.store(running_var2_ptr + channels2, new_var2, mask=channel_mask & ~first_bn)
    tl.store(mean2_ptr + channels2, mean, mask=channel_mask & ~first_bn)
    tl.store(invstd2_ptr + channels2, invstd, mask=channel_mask & ~first_bn)


@triton.jit
def _dual_bn_cat_add_kernel(
    x1_ptr,
    skip_ptr,
    x2_ptr,
    weight1_ptr,
    bias1_ptr,
    weight2_ptr,
    bias2_ptr,
    mean1_ptr,
    invstd1_ptr,
    mean2_ptr,
    invstd2_ptr,
    out_ptr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    stride1_n: tl.constexpr,
    stride1_c: tl.constexpr,
    stride1_h: tl.constexpr,
    stride1_w: tl.constexpr,
    stride_skip_n: tl.constexpr,
    stride_skip_c: tl.constexpr,
    stride_skip_h: tl.constexpr,
    stride_skip_w: tl.constexpr,
    stride2_n: tl.constexpr,
    stride2_c: tl.constexpr,
    stride2_h: tl.constexpr,
    stride2_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c2 = offsets % C2
    w_idx = (offsets // C2) % W
    h_idx = (offsets // (C2 * W)) % H
    n_idx = offsets // (C2 * H * W)

    x2_offsets = (
        n_idx * stride2_n
        + c2 * stride2_c
        + h_idx * stride2_h
        + w_idx * stride2_w
    )
    x2 = tl.load(x2_ptr + x2_offsets, mask=mask, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c2, mask=mask, other=0.0)
    invstd2 = tl.load(invstd2_ptr + c2, mask=mask, other=0.0)
    weight2 = tl.load(weight2_ptr + c2, mask=mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + c2, mask=mask, other=0.0).to(tl.float32)
    y2 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x2, mean2), invstd2), weight2), bias2)
    y2_bf16 = y2.to(tl.bfloat16)

    first_half = c2 < C1
    c1 = tl.where(first_half, c2, c2 - C1)
    x1_offsets = (
        n_idx * stride1_n
        + c1 * stride1_c
        + h_idx * stride1_h
        + w_idx * stride1_w
    )
    skip_offsets = (
        n_idx * stride_skip_n
        + c1 * stride_skip_c
        + h_idx * stride_skip_h
        + w_idx * stride_skip_w
    )
    skip = tl.load(skip_ptr + skip_offsets, mask=mask & first_half, other=0.0)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask & ~first_half, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c1, mask=mask & ~first_half, other=0.0)
    invstd1 = tl.load(invstd1_ptr + c1, mask=mask & ~first_half, other=0.0)
    weight1 = tl.load(weight1_ptr + c1, mask=mask & ~first_half, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c1, mask=mask & ~first_half, other=0.0).to(tl.float32)
    y1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x1, mean1), invstd1), weight1), bias1)
    y1_bf16 = y1.to(tl.bfloat16)
    cat_value = tl.where(first_half, skip, y1_bf16)
    out = (cat_value + y2_bf16).to(tl.bfloat16)

    out_offsets = (
        n_idx * out_stride_n
        + c2 * out_stride_c
        + h_idx * out_stride_h
        + w_idx * out_stride_w
    )
    tl.store(out_ptr + out_offsets, out, mask=mask)


# d46689c4: (T([512,80,7,7], bf16), T([80], f32), T([80], f32), T([80], f32), T([80], f32), T([512,80,7,7], bf16), T([512,160,7,7], bf16), T([160], f32), T([160], f32), T([160], f32), T([160], f32))
@oracle_impl(hardware="B200", point="d46689c4", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=8, OUT_BLOCK=1024, num_warps=4)
# ca0c1514: (T([512,56,14,14], bf16), T([56], f32), T([56], f32), T([56], f32), T([56], f32), T([512,56,14,14], bf16), T([512,112,14,14], bf16), T([112], f32), T([112], f32), T([112], f32), T([112], f32))
@oracle_impl(hardware="B200", point="ca0c1514", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=4, OUT_BLOCK=1024, num_warps=4)
# 5fcaf167: (T([512,40,14,14], bf16), T([40], f32), T([40], f32), T([40], f32), T([40], f32), T([512,40,14,14], bf16), T([512,80,14,14], bf16), T([80], f32), T([80], f32), T([80], f32), T([80], f32))
@oracle_impl(hardware="B200", point="5fcaf167", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=4, OUT_BLOCK=1024, num_warps=4)
# 7aa256e5: (T([512,20,28,28], bf16), T([20], f32), T([20], f32), T([20], f32), T([20], f32), T([512,20,28,28], bf16), T([512,40,28,28], bf16), T([40], f32), T([40], f32), T([40], f32), T([40], f32))
@oracle_impl(hardware="B200", point="7aa256e5", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=2, OUT_BLOCK=1024, num_warps=4)
# 3e95fdf8: (T([512,12,56,56], bf16), T([12], f32), T([12], f32), T([12], f32), T([12], f32), T([512,12,56,56], bf16), T([512,24,56,56], bf16), T([24], f32), T([24], f32), T([24], f32), T([24], f32))
@oracle_impl(hardware="B200", point="3e95fdf8", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=1, OUT_BLOCK=1024, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_E,
    C_BLOCK,
    FINAL_C_BLOCK,
    OUT_BLOCK,
    num_warps,
):
    (
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        skip,
        x2,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = inputs
    n = int(x1.shape[0])
    c1 = int(x1.shape[1])
    h = int(x1.shape[2])
    w = int(x1.shape[3])
    c2 = int(x2.shape[1])
    e = n * h * w
    total_c = c1 + c2
    num_chunks = triton.cdiv(e, BLOCK_E)
    block_chunks = triton.next_power_of_2(num_chunks)

    partial_sum = torch.empty((num_chunks, total_c), device=x1.device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_chunks, total_c), device=x1.device, dtype=torch.float32)
    mean1 = torch.empty_strided((1, c1, 1, 1), (c1, 1, 1, 1), device=x1.device, dtype=torch.float32)
    mean2 = torch.empty_strided((1, c2, 1, 1), (c2, 1, 1, 1), device=x1.device, dtype=torch.float32)
    invstd1 = torch.empty((c1,), device=x1.device, dtype=torch.float32)
    invstd2 = torch.empty((c2,), device=x1.device, dtype=torch.float32)
    out = torch.empty_strided(
        (n, c2, h, w),
        (c2 * h * w, 1, c2 * w, c2),
        device=x1.device,
        dtype=torch.bfloat16,
    )

    _dual_bn_partial_stats_kernel[(triton.cdiv(total_c, C_BLOCK), num_chunks)](
        x1,
        x2,
        partial_sum,
        partial_sum2,
        C1=c1,
        C2=c2,
        H=h,
        W=w,
        E=e,
        stride1_n=int(x1.stride(0)),
        stride1_c=int(x1.stride(1)),
        stride1_h=int(x1.stride(2)),
        stride1_w=int(x1.stride(3)),
        stride2_n=int(x2.stride(0)),
        stride2_c=int(x2.stride(1)),
        stride2_h=int(x2.stride(2)),
        stride2_w=int(x2.stride(3)),
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _dual_bn_finalize_stats_kernel[(triton.cdiv(total_c, FINAL_C_BLOCK),)](
        partial_sum,
        partial_sum2,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
        mean1,
        invstd1,
        mean2,
        invstd2,
        C1=c1,
        C2=c2,
        E=e,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=FINAL_C_BLOCK,
        num_warps=8 if block_chunks >= 1024 else 4,
        num_stages=3,
    )
    _dual_bn_cat_add_kernel[(triton.cdiv(out.numel(), OUT_BLOCK),)](
        x1,
        skip,
        x2,
        weight1,
        bias1,
        weight2,
        bias2,
        mean1,
        invstd1,
        mean2,
        invstd2,
        out,
        C1=c1,
        C2=c2,
        H=h,
        W=w,
        TOTAL=out.numel(),
        stride1_n=int(x1.stride(0)),
        stride1_c=int(x1.stride(1)),
        stride1_h=int(x1.stride(2)),
        stride1_w=int(x1.stride(3)),
        stride_skip_n=int(skip.stride(0)),
        stride_skip_c=int(skip.stride(1)),
        stride_skip_h=int(skip.stride(2)),
        stride_skip_w=int(skip.stride(3)),
        stride2_n=int(x2.stride(0)),
        stride2_c=int(x2.stride(1)),
        stride2_h=int(x2.stride(2)),
        stride2_w=int(x2.stride(3)),
        out_stride_n=int(out.stride(0)),
        out_stride_c=int(out.stride(1)),
        out_stride_h=int(out.stride(2)),
        out_stride_w=int(out.stride(3)),
        BLOCK=OUT_BLOCK,
        num_warps=4,
        num_stages=3,
    )

    return (
        invstd1,
        invstd2,
        out,
        mean2,
        mean1,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
    )
