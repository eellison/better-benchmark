"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 training-BatchNorm residual-add scope, including the fp32 population var_mean over N,H,W, eps=1e-5 rsqrt side output, mutable running-stat copy_ aliases with the captured variance-correction literal, affine bf16 cast, and returned bf16 residual add in the input layout, whereas Inductor lowers the reduction, running-stat updates, affine cast, and residual epilogue as generic schedules; Inductor cannot do this today because its scheduler does not emit a single guarded training-BatchNorm residual-add template that preserves mutable aliases, bf16 rounding boundaries, and channels-last or contiguous output strides; the fix is SCHEDULER_FUSION: extend the training-BatchNorm scheduler to fuse the channel-stat reduction, running-stat side effects, affine cast, and residual add into one full-scope lowering."""

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
def _bn_fused_small_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    invstd_ptr,
    y_ptr,
    mean_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    residual_s0: tl.constexpr,
    residual_s1: tl.constexpr,
    residual_s2: tl.constexpr,
    residual_s3: tl.constexpr,
    y_s0: tl.constexpr,
    y_s1: tl.constexpr,
    y_s2: tl.constexpr,
    y_s3: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = tl.arange(0, BLOCK_R)
    hw_size: tl.constexpr = H * W
    hw_idx = r_offsets % hw_size
    n_idx = r_offsets // hw_size
    h_idx = hw_idx // W
    w_idx = hw_idx - h_idx * W

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
    sum_x2 = tl.sum(x * x, axis=1)
    inv_e: tl.constexpr = 1.0 / E
    mean = sum_x * inv_e
    var = sum_x2 * inv_e - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + 1.0e-5)

    old_mean = tl.load(running_mean_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    new_mean = mean * 0.1 + old_mean * 0.9
    new_var = var * 1.0001594642002871 * 0.1 + old_var * 0.9

    tl.store(invstd_ptr + channels, invstd, mask=c_mask)
    tl.store(mean_ptr + channels, mean, mask=c_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=c_mask)
    tl.store(running_var_ptr + channels, new_var, mask=c_mask)

    weight = tl.load(weight_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    affine = ((x - mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
    rounded = affine.to(tl.bfloat16)

    residual_offsets = (
        n_idx[None, :] * residual_s0
        + channels[:, None] * residual_s1
        + h_idx[None, :] * residual_s2
        + w_idx[None, :] * residual_s3
    )
    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0)
    y = (rounded + residual).to(tl.bfloat16)

    y_offsets = (
        n_idx[None, :] * y_s0
        + channels[:, None] * y_s1
        + h_idx[None, :] * y_s2
        + w_idx[None, :] * y_s3
    )
    tl.store(y_ptr + y_offsets, y, mask=mask)


@triton.jit
def _bn_partial_stats_kernel(
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
    c_block = tl.program_id(0)
    chunk = tl.program_id(1)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = chunk * BLOCK_R + tl.arange(0, BLOCK_R)
    hw_size: tl.constexpr = H * W
    hw_idx = r_offsets % hw_size
    n_idx = r_offsets // hw_size
    h_idx = hw_idx // W
    w_idx = hw_idx - h_idx * W
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
    x = tl.where(mask, x, 0.0)
    partial_sum = tl.sum(x, axis=1)
    partial_sum2 = tl.sum(_f32_mul(x, x), axis=1)
    offsets = chunk * C + channels
    plane: tl.constexpr = NUM_CHUNKS * C
    tl.store(partial_ptr + offsets, partial_sum, mask=c_mask)
    tl.store(partial_ptr + plane + offsets, partial_sum2, mask=c_mask)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_ptr,
    running_mean_ptr,
    running_var_ptr,
    invstd_ptr,
    mean_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    c_mask = channels < C
    mask = c_mask[:, None] & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    plane: tl.constexpr = NUM_CHUNKS * C
    chunk_sum = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_sum2 = tl.load(partial_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(chunk_sum, axis=1)
    sum_x2 = tl.sum(chunk_sum2, axis=1)

    inv_e: tl.constexpr = 1.0 / E
    mean = _f32_mul(sum_x, inv_e)
    var = _f32_sub(_f32_mul(sum_x2, inv_e), _f32_mul(mean, mean))
    var = tl.where(var > 0.0, var, 0.0)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    new_var = _f32_add(
        _f32_mul(_f32_mul(var, 1.0001594642002871), 0.1),
        _f32_mul(old_var, 0.9),
    )

    tl.store(invstd_ptr + channels, invstd, mask=c_mask)
    tl.store(mean_ptr + channels, mean, mask=c_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=c_mask)
    tl.store(running_var_ptr + channels, new_var, mask=c_mask)


@triton.jit
def _bn_residual_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    invstd_ptr,
    mean_ptr,
    y_ptr,
    total_elements: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    if CHANNELS_LAST:
        channel = offsets % C
    else:
        channel = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    affine = _f32_add(_f32_mul(_f32_mul(_f32_sub(x, mean), invstd), weight), bias)
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    y = (rounded + residual).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(y_ptr + offsets, y, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@oracle_impl(hardware="B200", point="a23acbed", SMALL_BLOCK_C=8, SMALL_BLOCK_R=0, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=1, num_warps_epilogue=8)
@oracle_impl(hardware="B200", point="691fac28", SMALL_BLOCK_C=2, SMALL_BLOCK_R=32768, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="200ea7ae", SMALL_BLOCK_C=2, SMALL_BLOCK_R=32768, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="dadcc8c4", SMALL_BLOCK_C=1, SMALL_BLOCK_R=131072, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="f2c648cf", SMALL_BLOCK_C=1, SMALL_BLOCK_R=524288, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="c1e18312", SMALL_BLOCK_C=1, SMALL_BLOCK_R=32768, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="36791fa6", SMALL_BLOCK_C=1, SMALL_BLOCK_R=131072, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="b0c1fd9e", SMALL_BLOCK_C=1, SMALL_BLOCK_R=131072, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="bb24c64e", SMALL_BLOCK_C=1, SMALL_BLOCK_R=524288, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="70ec3859", SMALL_BLOCK_C=1, SMALL_BLOCK_R=2097152, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="1b3adb88", SMALL_BLOCK_C=1, SMALL_BLOCK_R=8388608, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="5f47b4f1", SMALL_BLOCK_C=1, SMALL_BLOCK_R=1048576, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="268ef9e2", SMALL_BLOCK_C=8, SMALL_BLOCK_R=0, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=1, num_warps_epilogue=8)
@oracle_impl(hardware="B200", point="9311a9c0", SMALL_BLOCK_C=2, SMALL_BLOCK_R=32768, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="8a557703", SMALL_BLOCK_C=2, SMALL_BLOCK_R=32768, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="96cd1bdd", SMALL_BLOCK_C=1, SMALL_BLOCK_R=131072, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="5cfa9308", SMALL_BLOCK_C=2, SMALL_BLOCK_R=2048, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=4, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="c69d9f2b", SMALL_BLOCK_C=2, SMALL_BLOCK_R=8192, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=4, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="7cea4043", SMALL_BLOCK_C=2, SMALL_BLOCK_R=8192, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=4, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="1e3428af", SMALL_BLOCK_C=2, SMALL_BLOCK_R=32768, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="dcb799ef", SMALL_BLOCK_C=1, SMALL_BLOCK_R=131072, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="007d658d", SMALL_BLOCK_C=2, SMALL_BLOCK_R=2048, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=4, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="16c9375e", SMALL_BLOCK_C=2, SMALL_BLOCK_R=8192, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=4, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="82eff7d5", SMALL_BLOCK_C=1, SMALL_BLOCK_R=524288, STATS_BLOCK_C=8, STATS_BLOCK_R=1024, EPILOGUE_BLOCK=1024, num_warps_small=8, num_warps_stats=4, num_warps_final=4, num_warps_epilogue=4)
def oracle_forward(
    inputs,
    *,
    SMALL_BLOCK_C: int,
    SMALL_BLOCK_R: int,
    STATS_BLOCK_C: int,
    STATS_BLOCK_R: int,
    EPILOGUE_BLOCK: int,
    num_warps_small: int,
    num_warps_stats: int,
    num_warps_final: int,
    num_warps_epilogue: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    batch, channels, height, width = (int(dim) for dim in arg0_1.shape)
    elements_per_channel = batch * height * width
    total_elements = batch * channels * height * width

    invstd = torch.empty_strided((channels,), (1,), device=arg0_1.device, dtype=torch.float32)
    y = torch.empty_strided(
        tuple(int(dim) for dim in arg0_1.shape),
        tuple(int(stride) for stride in arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided((1, channels, 1, 1), (channels, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)

    if elements_per_channel <= 8192 and elements_per_channel <= SMALL_BLOCK_R:
        _bn_fused_small_kernel[(triton.cdiv(channels, SMALL_BLOCK_C),)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            arg5_1,
            invstd,
            y,
            mean,
            N=batch,
            C=channels,
            H=height,
            W=width,
            E=elements_per_channel,
            x_s0=arg0_1.stride(0),
            x_s1=arg0_1.stride(1),
            x_s2=arg0_1.stride(2),
            x_s3=arg0_1.stride(3),
            residual_s0=arg5_1.stride(0),
            residual_s1=arg5_1.stride(1),
            residual_s2=arg5_1.stride(2),
            residual_s3=arg5_1.stride(3),
            y_s0=y.stride(0),
            y_s1=y.stride(1),
            y_s2=y.stride(2),
            y_s3=y.stride(3),
            BLOCK_C=SMALL_BLOCK_C,
            BLOCK_R=SMALL_BLOCK_R,
            num_warps=num_warps_small,
            num_stages=3,
        )
    else:
        num_chunks = triton.cdiv(elements_per_channel, STATS_BLOCK_R)
        block_chunks = _next_power_of_2(num_chunks)
        partial = torch.empty(
            (2, num_chunks, channels),
            device=arg0_1.device,
            dtype=torch.float32,
        )
        _bn_partial_stats_kernel[(triton.cdiv(channels, STATS_BLOCK_C), num_chunks)](
            arg0_1,
            partial,
            C=channels,
            H=height,
            W=width,
            E=elements_per_channel,
            x_s0=arg0_1.stride(0),
            x_s1=arg0_1.stride(1),
            x_s2=arg0_1.stride(2),
            x_s3=arg0_1.stride(3),
            NUM_CHUNKS=num_chunks,
            BLOCK_C=STATS_BLOCK_C,
            BLOCK_R=STATS_BLOCK_R,
            num_warps=num_warps_stats,
            num_stages=3,
        )
        _bn_finalize_stats_kernel[(triton.cdiv(channels, STATS_BLOCK_C),)](
            partial,
            arg1_1,
            arg2_1,
            invstd,
            mean,
            C=channels,
            E=elements_per_channel,
            NUM_CHUNKS=num_chunks,
            BLOCK_C=STATS_BLOCK_C,
            BLOCK_CHUNKS=block_chunks,
            num_warps=num_warps_final,
            num_stages=3,
        )
        _bn_residual_kernel[(triton.cdiv(total_elements, EPILOGUE_BLOCK),)](
            arg0_1,
            arg3_1,
            arg4_1,
            arg5_1,
            invstd,
            mean,
            y,
            total_elements=total_elements,
            C=channels,
            HW=height * width,
            CHANNELS_LAST=arg0_1.stride(1) == 1,
            BLOCK=EPILOGUE_BLOCK,
            num_warps=num_warps_epilogue,
            num_stages=3,
        )

    return invstd, y, mean, arg1_1, arg2_1
