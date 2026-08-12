"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Shufflenet bf16 training-BatchNorm plus skip-branch channel-shuffle scope, including fp32 var_mean over the bf16 activation, mean and rsqrt side outputs, running mean/variance copy_ side effects, the affine epilogue with the explicit bf16 cast before ReLU, cat/view/permute/clone/view layout transform, and split view returns by writing the skip branch and BN branch directly into the final shuffled storage, whereas Inductor lowers the decomposed BN, mutable running-stat updates, concat, permutation clone, and split as generic reduction and layout work; Inductor cannot do this today because its scheduler does not preserve the fixed channel-shuffle consumer across a training normalization producer with side-effect outputs and an already-strided sibling input; the fix is SCHEDULER_FUSION: teach the BN-training/layout scheduler to fuse running-stat epilogues with direct channel-shuffle stores while preserving side outputs, bf16 rounding boundaries, and split-view aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 116
H = 14
W = 14
HW = H * W
K = N * HW
OUT_C = C * 2
EPS = 1.0e-5
MOMENTUM = 0.1
OLD_WEIGHT = 0.9
RUNNING_VAR_CORRECTION = 1.0000398612827361


@triton.jit
def _bn_stats_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    k_size: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    old_weight: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < k_size
    n_idx = offsets // hw_size
    hw_idx = offsets - n_idx * hw_size

    x_offsets = n_idx * channels * hw_size + channel * hw_size + hw_idx
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / k_size
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / k_size
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + eps)

    tl.store(mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(running_mean_ptr + channel, old_mean * old_weight + mean * momentum)
    tl.store(
        running_var_ptr + channel,
        old_var * old_weight + var * running_var_correction * momentum,
    )


@triton.jit
def _shuffle_output_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    skip_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    total: tl.constexpr,
    channels: tl.constexpr,
    out_channels: tl.constexpr,
    hw_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    hw_idx = offsets % hw_size
    channel = (offsets // hw_size) % channels
    n_idx = offsets // (channels * hw_size)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * invstd * weight + bias
    y_rounded = y.to(tl.bfloat16).to(tl.float32)
    y_relu = tl.where(y_rounded != y_rounded, y_rounded, tl.maximum(y_rounded, 0.0))

    skip_offsets = n_idx * out_channels * hw_size + channel * hw_size + hw_idx
    skip = tl.load(skip_ptr + skip_offsets, mask=mask, other=0.0)

    out_base = n_idx * out_channels * hw_size + channel * 2 * hw_size + hw_idx
    tl.store(out_ptr + out_base, skip, mask=mask)
    tl.store(out_ptr + out_base + hw_size, y_relu, mask=mask)


# 00e34431: (T([128,116,14,14], bf16), T([116], f32), ..., strided skip branch)
@oracle_impl(hardware="B200", point="00e34431", BLOCK_K=32768, BLOCK_OUT=256)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_OUT: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1 = inputs

    mean = torch.empty_strided(
        (1, C, 1, 1),
        (C, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, C, 1, 1),
        (C, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    shuffled = torch.empty_strided(
        (N, OUT_C, H, W),
        (OUT_C * HW, HW, W, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _bn_stats_kernel[(C,)](
        arg0_1,
        arg1_1,
        arg2_1,
        mean,
        invstd,
        channels=C,
        hw_size=HW,
        k_size=K,
        eps=EPS,
        momentum=MOMENTUM,
        old_weight=OLD_WEIGHT,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    _shuffle_output_kernel[(triton.cdiv(N * C * HW, BLOCK_OUT),)](
        arg0_1,
        arg3_1,
        arg4_1,
        arg5_1,
        mean,
        invstd,
        shuffled,
        total=N * C * HW,
        channels=C,
        out_channels=OUT_C,
        hw_size=HW,
        BLOCK=BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )

    return (
        mean,
        invstd,
        shuffled[:, :C, :, :],
        shuffled[:, C:, :, :],
        arg1_1,
        arg2_1,
    )
