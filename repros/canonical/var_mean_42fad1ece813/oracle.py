"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 training-BatchNorm affine ReLU scope, including fp32 population var_mean over `[N,H,W]`, eps=0.001 rsqrt, `[C]` invstd output, `[1,C,1,1]` saved-mean output, in-place running mean/variance copy_ updates with the captured fixed correction constant, bf16 rounding before ReLU, and the full activation store in the input's dense memory format, whereas Inductor lowers the canonicalized var_mean normalization graph through generic norm-template and pointwise schedules; Inductor cannot do this today because the scheduler does not keep the BN statistics, mutable running-stat epilogues, affine, bf16 cast, and ReLU in one compact full-scope plan across the recorded contiguous and channels-last shapes; the fix is SCHEDULER_FUSION: extend the training BatchNorm norm template to emit the side outputs, running-stat aliases, cast boundary, and fused affine/ReLU epilogue directly from the channel-statistics schedule when profitable."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bn_partial_stats_kernel(
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
    sums = tl.sum(vals, axis=1)
    sums2 = tl.sum(vals * vals, axis=1)
    out_offsets = e_block * C + channels
    tl.store(partial_sum_ptr + out_offsets, sums, mask=channels < C)
    tl.store(partial_sum2_ptr + out_offsets, sums2, mask=channels < C)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_sum_ptr,
    partial_sum2_ptr,
    running_mean_ptr,
    running_var_ptr,
    invstd_ptr,
    mean_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)
    sums2 = tl.load(partial_sum2_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(sums, axis=1).to(tl.float32)
    total2 = tl.sum(sums2, axis=1).to(tl.float32)
    mean = total / E
    var = total2 / E - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + 0.001)
    channel_mask = channels < C

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    tl.store(running_mean_ptr + channels, old_mean * 0.9 + mean * 0.1, mask=channel_mask)
    tl.store(
        running_var_ptr + channels,
        old_var * 0.9 + var * 1.0001220852154804 * 0.1,
        mask=channel_mask,
    )
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(mean_ptr + channels, mean, mask=channel_mask)


@triton.jit
def _bn_affine_relu_kernel(
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
    y = y.to(tl.bfloat16)
    y = tl.maximum(y, 0.0)
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
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = tl.arange(0, BLOCK_E)
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
    sums = tl.sum(vals, axis=1)
    sums2 = tl.sum(vals * vals, axis=1)
    mean = sums / E
    var = sums2 / E - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + 0.001)
    channel_mask = channels < C

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    tl.store(running_mean_ptr + channels, old_mean * 0.9 + mean * 0.1, mask=channel_mask)
    tl.store(
        running_var_ptr + channels,
        old_var * 0.9 + var * 1.0001220852154804 * 0.1,
        mask=channel_mask,
    )
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(mean_ptr + channels, mean, mask=channel_mask)

    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    y = ((vals - mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
    y = y.to(tl.bfloat16)
    y = tl.maximum(y, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="652894ae")
@oracle_impl(hardware="B200", point="8e6a7889")
@oracle_impl(hardware="B200", point="5921a507")
@oracle_impl(hardware="B200", point="bdbc3e2e")
@oracle_impl(hardware="B200", point="2534b995")
@oracle_impl(hardware="B200", point="8913da8d")
@oracle_impl(hardware="B200", point="371d495a")
@oracle_impl(hardware="B200", point="cf977d52")
@oracle_impl(hardware="B200", point="9514c47f")
@oracle_impl(hardware="B200", point="0d2cf849")
@oracle_impl(hardware="B200", point="bf947a21")
@oracle_impl(hardware="B200", point="4b83d612")
@oracle_impl(hardware="B200", point="d19df2e3")
@oracle_impl(hardware="B200", point="df1d49cf")
@oracle_impl(hardware="B200", point="0573e23e")
@oracle_impl(hardware="B200", point="ef578a31")
@oracle_impl(hardware="B200", point="765e4345")
@oracle_impl(hardware="B200", point="d49692af")
@oracle_impl(hardware="B200", point="829d7a6e")
@oracle_impl(hardware="B200", point="abb568e6")
@oracle_impl(hardware="B200", point="ce954876")
@oracle_impl(hardware="B200", point="a3c80480")
@oracle_impl(hardware="B200", point="076257b8")
@oracle_impl(hardware="B200", point="f01b9cb5")
@oracle_impl(hardware="B200", point="916ac6ee")
@oracle_impl(hardware="B200", point="bf1cc8fb")
@oracle_impl(hardware="B200", point="49824dd5")
@oracle_impl(hardware="B200", point="3798dfee")
@oracle_impl(hardware="B200", point="67dcaf05")
@oracle_impl(hardware="B200", point="7feb8094")
@oracle_impl(hardware="B200", point="2448297b")
@oracle_impl(hardware="B200", point="689c8296")
@oracle_impl(hardware="B200", point="159050d6")
@oracle_impl(hardware="B200", point="ce1d154b")
@oracle_impl(hardware="B200", point="ba044ce1")
@oracle_impl(hardware="B200", point="4e453d00")
@oracle_impl(hardware="B200", point="85aaacf2")
@oracle_impl(hardware="B200", point="00cca0ad")
@oracle_impl(hardware="B200", point="b7fea7d6")
@oracle_impl(hardware="B200", point="e85936d9")
@oracle_impl(hardware="B200", point="e10b7d9b")
@oracle_impl(hardware="B200", point="4d8049ba")
@oracle_impl(hardware="B200", point="7b93f272")
@oracle_impl(hardware="B200", point="7dc93580")
@oracle_impl(hardware="B200", point="22d6d5c7")
@oracle_impl(hardware="B200", point="5e2cd32b")
@oracle_impl(hardware="B200", point="e10c3aee")
@oracle_impl(hardware="B200", point="c0fff287")
@oracle_impl(hardware="B200", point="668a6e2e")
@oracle_impl(hardware="B200", point="2777319d")
@oracle_impl(hardware="B200", point="a084d359")
@oracle_impl(hardware="B200", point="01afa305")
@oracle_impl(hardware="B200", point="c31644f9")
@oracle_impl(hardware="B200", point="baaf45aa")
@oracle_impl(hardware="B200", point="b0d2ca15")
@oracle_impl(hardware="B200", point="18f77a77")
@oracle_impl(hardware="B200", point="27477e9d")
@oracle_impl(hardware="B200", point="91e03843")
@oracle_impl(hardware="B200", point="5afa3f59")
@oracle_impl(hardware="B200", point="fd34a6df")
@oracle_impl(hardware="B200", point="dcfe1ef6")
@oracle_impl(hardware="B200", point="6380a151")
@oracle_impl(hardware="B200", point="de255c00")
@oracle_impl(hardware="B200", point="301a86d7")
@oracle_impl(hardware="B200", point="e2a288dc")
@oracle_impl(hardware="B200", point="16f343e4")
@oracle_impl(hardware="B200", point="0556e565")
@oracle_impl(hardware="B200", point="899450c5")
@oracle_impl(hardware="B200", point="580ef142")
@oracle_impl(hardware="B200", point="0c92aee1")
@oracle_impl(hardware="B200", point="3b535b42")
@oracle_impl(hardware="B200", point="df8b8e1c")
@oracle_impl(hardware="B200", point="30ad209f")
@oracle_impl(hardware="B200", point="da574aef")
@oracle_impl(hardware="B200", point="d3755540")
@oracle_impl(hardware="B200", point="5d75103c")
@oracle_impl(hardware="B200", point="10267491")
@oracle_impl(hardware="B200", point="f3f3df2a")
@oracle_impl(hardware="B200", point="09d0dd48")
@oracle_impl(hardware="B200", point="04752777")
@oracle_impl(hardware="B200", point="33eef2b0")
@oracle_impl(hardware="B200", point="d37f9b8e")
@oracle_impl(hardware="B200", point="1c964cfd")
@oracle_impl(hardware="B200", point="57daf190")
@oracle_impl(hardware="B200", point="5c409da5")
@oracle_impl(hardware="B200", point="0ba0f4c1")
@oracle_impl(hardware="B200", point="bff20525")
@oracle_impl(hardware="B200", point="6627cb81")
@oracle_impl(hardware="B200", point="ec1aa18f")
@oracle_impl(hardware="B200", point="82614c26")
@oracle_impl(hardware="B200", point="d8555100")
@oracle_impl(hardware="B200", point="6ca1c7fb")
@oracle_impl(hardware="B200", point="9bfd45b0")
@oracle_impl(hardware="B200", point="f3b02708")
@oracle_impl(hardware="B200", point="f002df5c")
@oracle_impl(hardware="B200", point="c8b4a4a9")
@oracle_impl(hardware="B200", point="98955048")
@oracle_impl(hardware="B200", point="21cf057f")
@oracle_impl(hardware="B200", point="9e46e7dc")
@oracle_impl(hardware="B200", point="8cbf400d")
@oracle_impl(hardware="B200", point="138137a0")
@oracle_impl(hardware="B200", point="d1ba5867")
@oracle_impl(hardware="B200", point="b6bdfd5b")
@oracle_impl(hardware="B200", point="37819cd3")
@oracle_impl(hardware="B200", point="538399db")
@oracle_impl(hardware="B200", point="8e0806dd")
@oracle_impl(hardware="B200", point="6750c5ad")
@oracle_impl(hardware="B200", point="1374dde1")
@oracle_impl(hardware="B200", point="a1b0f35d")
@oracle_impl(hardware="B200", point="2f268f5e")
@oracle_impl(hardware="B200", point="0422d4bd")
@oracle_impl(hardware="B200", point="35112db7")
@oracle_impl(hardware="B200", point="d8517324")
@oracle_impl(hardware="B200", point="f3e6b4a8")
@oracle_impl(hardware="B200", point="a35361ae")
@oracle_impl(hardware="B200", point="25fd553d")
@oracle_impl(hardware="B200", point="b099d844")
@oracle_impl(hardware="B200", point="3017b2d7")
@oracle_impl(hardware="B200", point="3cd4808f")
@oracle_impl(hardware="B200", point="512285fa")
@oracle_impl(hardware="B200", point="530fd4f8")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias = inputs
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    e = n * hw
    total = n * c * hw

    block_e = 1024
    if e <= 256:
        block_e = triton.next_power_of_2(e)
    elif e <= 512:
        block_e = 512
    c_block = 16 if e <= 1024 else 8
    num_chunks = triton.cdiv(e, block_e)
    block_chunks = triton.next_power_of_2(num_chunks)

    invstd = torch.empty_strided((c,), (1,), device=x.device, dtype=torch.float32)
    relu = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)

    if e <= 1024:
        block_e = triton.next_power_of_2(e)
        c_block = 1 if e <= 256 else 16
        _bn_single_pass_kernel[(triton.cdiv(c, c_block),)](
            x,
            running_mean,
            running_var,
            weight,
            bias,
            invstd,
            relu,
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
            C_BLOCK=c_block,
            num_warps=1 if e <= 256 else (8 if c_block * block_e >= 8192 else 4),
        )
        return invstd, relu, mean, running_mean, running_var

    partial_sum = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)

    _bn_partial_stats_kernel[(triton.cdiv(c, c_block), num_chunks)](
        x,
        partial_sum,
        partial_sum2,
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
    _bn_finalize_stats_kernel[(triton.cdiv(c, c_block),)](
        partial_sum,
        partial_sum2,
        running_mean,
        running_var,
        invstd,
        mean,
        c,
        e,
        num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=c_block,
        num_warps=8 if c_block * block_chunks >= 8192 else 4,
    )
    _bn_affine_relu_kernel[(triton.cdiv(total, 1024),)](
        x,
        weight,
        bias,
        invstd,
        mean,
        relu,
        c,
        hw,
        total,
        CHANNELS_LAST=x.stride(1) == 1,
        BLOCK=1024,
        num_warps=4,
    )
    return invstd, relu, mean, running_mean, running_var
