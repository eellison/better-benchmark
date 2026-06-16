"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 training-BatchNorm forward scope, including the input bf16-to-fp32 cast, population var_mean over `[N,H,W]`, eps=1e-5 rsqrt, bf16 affine-normalized output, `[C]` invstd output, `[1,C,1,1]` saved-mean output, and both in-place running-stat copy_ updates with the captured correction constant, by splitting each per-channel statistics reduction into cooperative chunks before a compact finalizer and dense affine epilogue, whereas Inductor lowers this canonicalized norm-template graph as generic statistics and pointwise schedules; Inductor cannot do this today because the normalization scheduler lacks a guarded cooperative split-K training-BatchNorm variant that also preserves mutable running-stat aliases and side outputs across both contiguous and channels-last dense layouts; the fix is COOPERATIVE_SPLIT_K: add a BN-training stats template that splits the reduction dimension, finalizes invstd/mean/running-stat side outputs, and feeds the affine bf16 store from the same schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bn_partial_stats_kernel(
    x_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    e_block = tl.program_id(1)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = e_block * BLOCK_E + tl.arange(0, BLOCK_E)
    hw = e_offsets % (H * W)
    n_idx = e_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.where(mask, 1.0, 0.0)
    block_mean, block_m2, _ = triton_helpers.welford(
        vals,
        tl.zeros([C_BLOCK, BLOCK_E], tl.float32),
        weights,
        1,
    )
    out_offsets = e_block * C + channels
    tl.store(partial_mean_ptr + out_offsets, block_mean, mask=channels < C)
    tl.store(partial_m2_ptr + out_offsets, block_m2, mask=channels < C)


@triton.jit
def _bn_partial_stats_flat_cl_kernel(
    x_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    R_BLOCK: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    x_offsets = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)
    channels = x_offsets % C
    chunks = x_offsets // C
    r_base = tl.arange(0, R_BLOCK)
    mean_acc = tl.zeros([X_BLOCK, R_BLOCK], tl.float32)
    m2_acc = tl.zeros([X_BLOCK, R_BLOCK], tl.float32)
    weight_acc = tl.zeros([X_BLOCK, R_BLOCK], tl.float32)
    for r_offset in tl.range(0, BLOCK_E, R_BLOCK):
        r_offsets = r_offset + r_base
        e_offsets = chunks[:, None] * BLOCK_E + r_offsets[None, :]
        offsets = channels[:, None] + C * e_offsets
        mask = (x_offsets[:, None] < C * tl.cdiv(E, BLOCK_E)) & (e_offsets < E)
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            vals,
            mean_acc,
            m2_acc,
            weight_acc,
            r_offset == 0,
        )
        mean_acc = tl.where(mask, mean_next, mean_acc)
        m2_acc = tl.where(mask, m2_next, m2_acc)
        weight_acc = tl.where(mask, weight_next, weight_acc)
    mean, m2, _ = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    tl.store(partial_mean_ptr + x_offsets, mean, mask=x_offsets < C * tl.cdiv(E, BLOCK_E))
    tl.store(partial_m2_ptr + x_offsets, m2, mask=x_offsets < C * tl.cdiv(E, BLOCK_E))


@triton.jit
def _bn_combine_stats_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    grouped_mean_ptr,
    grouped_m2_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    GROUP_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    group = tl.program_id(1)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = group * GROUP_CHUNKS + tl.arange(0, GROUP_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    chunk_start = chunks * BLOCK_E
    counts = tl.where(chunks < NUM_CHUNKS, tl.minimum(BLOCK_E, E - chunk_start), 0).to(tl.float32)
    block_mean = tl.load(partial_mean_ptr + offsets, mask=mask, other=0.0)
    block_m2 = tl.load(partial_m2_ptr + offsets, mask=mask, other=0.0)
    weights = tl.broadcast_to(counts[None, :], [C_BLOCK, GROUP_CHUNKS])
    zero = tl.zeros([C_BLOCK, GROUP_CHUNKS], tl.float32)
    combined_mean, combined_m2, combined_weight = triton_helpers.welford_combine(
        zero,
        zero,
        zero,
        block_mean,
        block_m2,
        weights,
    )
    mean, m2, _ = triton_helpers.welford(combined_mean, combined_m2, combined_weight, 1)
    out_offsets = group * C + channels
    tl.store(grouped_mean_ptr + out_offsets, mean, mask=channels < C)
    tl.store(grouped_m2_ptr + out_offsets, m2, mask=channels < C)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    running_mean_ptr,
    running_var_ptr,
    invstd_ptr,
    mean_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    chunk_start = chunks * BLOCK_E
    counts = tl.where(chunks < NUM_CHUNKS, tl.minimum(BLOCK_E, E - chunk_start), 0).to(tl.float32)
    block_mean = tl.load(partial_mean_ptr + offsets, mask=mask, other=0.0)
    block_m2 = tl.load(partial_m2_ptr + offsets, mask=mask, other=0.0)
    weights = tl.broadcast_to(counts[None, :], [C_BLOCK, BLOCK_CHUNKS])
    mean, m2, _ = triton_helpers.welford(block_mean, block_m2, weights, 1)
    var = m2 / E
    invstd = libdevice.rsqrt(var + 1.0e-5)
    channel_mask = channels < C

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    tl.store(running_mean_ptr + channels, old_mean * 0.9 + mean * 0.1, mask=channel_mask)
    tl.store(
        running_var_ptr + channels,
        old_var * 0.9 + var * 1.0000398612827361 * 0.1,
        mask=channel_mask,
    )
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(mean_ptr + channels, mean, mask=channel_mask)


@triton.jit
def _bn_affine_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    invstd_ptr,
    mean_ptr,
    y_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    if CHANNELS_LAST:
        channels = offsets % C
    else:
        channels = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0)
    invstd = tl.load(invstd_ptr + channels, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    y = ((x - mean) * invstd) * weight + bias
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _bn_single_pass_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    invstd_ptr,
    y_ptr,
    mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_E: tl.constexpr,
    R_BLOCK: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    r_base = tl.arange(0, R_BLOCK)
    mean_acc = tl.zeros([C_BLOCK, R_BLOCK], tl.float32)
    m2_acc = tl.zeros([C_BLOCK, R_BLOCK], tl.float32)
    weight_acc = tl.zeros([C_BLOCK, R_BLOCK], tl.float32)
    for r_offset in tl.range(0, BLOCK_E, R_BLOCK):
        e_offsets = r_offset + r_base
        hw = e_offsets % (H * W)
        n_idx = e_offsets // (H * W)
        h_idx = hw // W
        w_idx = hw - h_idx * W
        offsets = (
            n_idx[None, :] * stride_n
            + channels[:, None] * stride_c
            + h_idx[None, :] * stride_h
            + w_idx[None, :] * stride_w
        )
        mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            vals,
            mean_acc,
            m2_acc,
            weight_acc,
            r_offset == 0,
        )
        mean_acc = tl.where(mask, mean_next, mean_acc)
        m2_acc = tl.where(mask, m2_next, m2_acc)
        weight_acc = tl.where(mask, weight_next, weight_acc)
    mean, m2, _ = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    var = m2 / E
    invstd = libdevice.rsqrt(var + 1.0e-5)
    channel_mask = channels < C

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    tl.store(running_mean_ptr + channels, old_mean * 0.9 + mean * 0.1, mask=channel_mask)
    tl.store(
        running_var_ptr + channels,
        old_var * 0.9 + var * 1.0000398612827361 * 0.1,
        mask=channel_mask,
    )
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(mean_ptr + channels, mean, mask=channel_mask)

    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    for r_offset in tl.range(0, BLOCK_E, R_BLOCK):
        e_offsets = r_offset + r_base
        hw = e_offsets % (H * W)
        n_idx = e_offsets // (H * W)
        h_idx = hw // W
        w_idx = hw - h_idx * W
        offsets = (
            n_idx[None, :] * stride_n
            + channels[:, None] * stride_c
            + h_idx[None, :] * stride_h
            + w_idx[None, :] * stride_w
        )
        mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = ((vals - mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
        tl.store(y_ptr + offsets, y, mask=mask)

@oracle_impl(hardware="B200", point="c2a1e3e9")
@oracle_impl(hardware="B200", point="7db2aeb2")
@oracle_impl(hardware="B200", point="31096850")
@oracle_impl(hardware="B200", point="6f172941")
@oracle_impl(hardware="B200", point="17ef9ad6")
@oracle_impl(hardware="B200", point="df1d49cf")
@oracle_impl(hardware="B200", point="dd3c979f")
@oracle_impl(hardware="B200", point="17c54342")
@oracle_impl(hardware="B200", point="2da7ced3")
@oracle_impl(hardware="B200", point="a897ff09")
@oracle_impl(hardware="B200", point="6058e15b")
@oracle_impl(hardware="B200", point="ce954876")
@oracle_impl(hardware="B200", point="a8a74e41")
@oracle_impl(hardware="B200", point="97fb8c65")
@oracle_impl(hardware="B200", point="62c751f1")
@oracle_impl(hardware="B200", point="a20aa279")
@oracle_impl(hardware="B200", point="b5e847de")
@oracle_impl(hardware="B200", point="5c7c7725")
@oracle_impl(hardware="B200", point="75c62b73")
@oracle_impl(hardware="B200", point="40783eff")
@oracle_impl(hardware="B200", point="2a6b1005")
@oracle_impl(hardware="B200", point="1ddb5b13")
@oracle_impl(hardware="B200", point="87cb2ffa")
@oracle_impl(hardware="B200", point="000da78a")
@oracle_impl(hardware="B200", point="3a7a99d6")
@oracle_impl(hardware="B200", point="ddcb5e92")
@oracle_impl(hardware="B200", point="004f43c9")
@oracle_impl(hardware="B200", point="98b145fe")
@oracle_impl(hardware="B200", point="0086f84b")
@oracle_impl(hardware="B200", point="5591e6a1")
@oracle_impl(hardware="B200", point="1261775f")
@oracle_impl(hardware="B200", point="cb33c3d1")
@oracle_impl(hardware="B200", point="d0ac0091")
@oracle_impl(hardware="B200", point="0cca0118")
@oracle_impl(hardware="B200", point="f9264590")
@oracle_impl(hardware="B200", point="b3e3f5ef")
@oracle_impl(hardware="B200", point="1a8507ca")
@oracle_impl(hardware="B200", point="b9d86480")
@oracle_impl(hardware="B200", point="5e2cd32b")
@oracle_impl(hardware="B200", point="126e37ab")
@oracle_impl(hardware="B200", point="a4f815c2")
@oracle_impl(hardware="B200", point="b76cea4d")
@oracle_impl(hardware="B200", point="df1dfc57")
@oracle_impl(hardware="B200", point="7a0912ed")
@oracle_impl(hardware="B200", point="d7e99106")
@oracle_impl(hardware="B200", point="46eca283")
@oracle_impl(hardware="B200", point="0fbf9a67")
@oracle_impl(hardware="B200", point="fc44e6c2")
@oracle_impl(hardware="B200", point="21cf057f")
@oracle_impl(hardware="B200", point="8cbf400d")
@oracle_impl(hardware="B200", point="d1ba5867")
@oracle_impl(hardware="B200", point="e031939e")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias = inputs
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    e = n * hw
    total = n * c * hw

    block_e = 2048
    if e <= 256:
        block_e = triton.next_power_of_2(e)
    elif e <= 512:
        block_e = 512
    elif e <= 2048:
        block_e = 2048
    elif n == 128 and c == 32 and h == 128 and w == 128:
        block_e = 2048
    elif e <= 32768:
        block_e = 4096 if (n == 128 and c == 64 and h == 14 and w == 14) else 1024
    else:
        block_e = max(512, triton.next_power_of_2(triton.cdiv(e, 1024)))
    c_block = 16 if e <= 2048 else 8
    num_chunks = triton.cdiv(e, block_e)
    block_chunks = triton.next_power_of_2(num_chunks)

    invstd = torch.empty_strided((c,), (1,), device=x.device, dtype=torch.float32)
    y = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)

    if e <= 2048:
        block_e = triton.next_power_of_2(e)
        if n == 32 and h == 7 and w == 7:
            c_block = 1 if c == 320 else 4
            r_block = block_e if c == 320 else 512
        else:
            c_block = 1 if e <= 256 else 16
            r_block = block_e if e <= 256 else 512
        num_warps = 1 if e <= 256 else (8 if c_block * block_e >= 8192 else 4)
        _bn_single_pass_kernel[(triton.cdiv(c, c_block),)](
            x,
            running_mean,
            running_var,
            weight,
            bias,
            invstd,
            y,
            mean,
            c,
            h,
            w,
            e,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            BLOCK_E=block_e,
            R_BLOCK=r_block,
            C_BLOCK=c_block,
            num_warps=num_warps,
        )
        return invstd, y, mean, running_mean, running_var

    partial_mean = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    partial_m2 = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)

    use_inductor_tree = n == 128 and c == 32 and h == 128 and w == 128
    if use_inductor_tree:
        _bn_partial_stats_flat_cl_kernel[(c * num_chunks,)](
            x,
            partial_mean,
            partial_m2,
            c,
            e,
            BLOCK_E=block_e,
            R_BLOCK=256,
            X_BLOCK=1,
            num_warps=8,
        )
    else:
        _bn_partial_stats_kernel[(triton.cdiv(c, c_block), num_chunks)](
            x,
            partial_mean,
            partial_m2,
            c,
            h,
            w,
            e,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            BLOCK_E=block_e,
            C_BLOCK=c_block,
            num_warps=8 if c_block * block_e >= 8192 else 4,
        )
    if use_inductor_tree:
        group_chunks = 128
        grouped_chunks = triton.cdiv(num_chunks, group_chunks)
        grouped_mean = torch.empty((grouped_chunks, c), device=x.device, dtype=torch.float32)
        grouped_m2 = torch.empty((grouped_chunks, c), device=x.device, dtype=torch.float32)
        _bn_combine_stats_kernel[(triton.cdiv(c, c_block), grouped_chunks)](
            partial_mean,
            partial_m2,
            grouped_mean,
            grouped_m2,
            c,
            e,
            num_chunks,
            BLOCK_E=block_e,
            GROUP_CHUNKS=group_chunks,
            C_BLOCK=c_block,
            num_warps=8 if c_block * group_chunks >= 8192 else 4,
        )
        partial_mean = grouped_mean
        partial_m2 = grouped_m2
        num_chunks = grouped_chunks
        block_e = block_e * group_chunks
        block_chunks = triton.next_power_of_2(num_chunks)
    _bn_finalize_stats_kernel[(triton.cdiv(c, c_block),)](
        partial_mean,
        partial_m2,
        running_mean,
        running_var,
        invstd,
        mean,
        c,
        e,
        num_chunks,
        BLOCK_E=block_e,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=c_block,
        num_warps=8 if c_block * block_chunks >= 8192 else 4,
    )
    _bn_affine_kernel[(triton.cdiv(total, 1024),)](
        x,
        weight,
        bias,
        invstd,
        mean,
        y,
        c,
        hw,
        total,
        CHANNELS_LAST=x.stride(1) == 1,
        BLOCK=1024,
        num_warps=4,
    )
    return invstd, y, mean, running_mean, running_var
