"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 training-BatchNorm plus SiLU spatial-mean scope with channel-specialized Triton kernels, including channels-last input loads, population `var_mean(correction=0)` over `[N,H,W]`, eps=1e-5 `rsqrt`, running mean/variance `copy_` side effects with unbiased running-var correction, affine normalization, observable bf16 cast before `exp`, natural-exp SiLU, observable bf16 cast before the spatial mean, returned mean/rsqrt side tensors, final `[N,C]` bf16 view, and returned running-stat aliases, whereas Inductor lowers the BatchNorm statistics, mutable running-stat updates, affine/SiLU producer, and downstream spatial mean through generic reduction scheduling; Inductor cannot do this today because the scheduler does not keep training-normalization side effects and the nonlinear spatial-mean consumer inside one fixed channels-last per-channel plan; the fix is SCHEDULER_FUSION: teach the BN-training norm lowering to expose running-stat epilogues while fusing affine/activation spatial reductions and side-output stores into one schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _fused_bn_silu_spatial_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    mean_out_ptr,
    rsqrt_out_ptr,
    spatial_out_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    channel = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)[:, None]
    hw = tl.arange(0, BLOCK_HW)[None, :]
    h = hw // width
    w = hw - h * width
    mask = hw < hw_size

    offsets = n * channels * hw_size + channel + h * width * channels + w * channels
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    vals = tl.where(mask, vals, 0.0)

    row_sum = tl.sum(vals, axis=1)
    row_sq_sum = tl.sum(vals * vals, axis=1)
    total = tl.sum(row_sum, axis=0)
    total_sq = tl.sum(row_sq_sum, axis=0)
    mean = total / elements_per_channel
    variance = total_sq / elements_per_channel - mean * mean
    variance = tl.maximum(variance, 0.0)
    invstd = tl.rsqrt(variance + 1.0e-5)

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(running_mean_ptr + channel, old_mean * 0.9 + mean * 0.1)
    tl.store(
        running_var_ptr + channel,
        old_var * 0.9 + variance * running_var_correction * 0.1,
    )
    tl.store(mean_out_ptr + channel, mean)
    tl.store(rsqrt_out_ptr + channel, invstd)

    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    y = ((vals - mean) * invstd) * weight + bias
    y = y.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    silu = y / (libdevice.exp(-y) + 1.0)
    silu = silu.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    spatial_sum = tl.sum(tl.where(mask, silu, 0.0), axis=1)
    spatial_mean = spatial_sum / hw_size
    tl.store(
        spatial_out_ptr + tl.arange(0, BLOCK_N) * channels + channel,
        spatial_mean.to(tl.bfloat16, fp_downcast_rounding="rtne"),
    )


def _launch(
    inputs,
    *,
    CHANNELS: int,
    HEIGHT: int,
    WIDTH: int,
    BLOCK_N: int,
    BLOCK_HW: int,
    running_var_correction: float,
    num_warps: int,
    num_stages: int,
):
    x, running_mean, running_var, weight, bias, _shape0, _shape1, shape2 = inputs
    batch = int(x.shape[0])
    hw_size = HEIGHT * WIDTH
    mean_out = torch.empty_strided(
        (1, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    rsqrt_out = torch.empty_strided(
        (1, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    spatial_out = torch.empty_strided(
        tuple(int(dim) for dim in shape2),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _fused_bn_silu_spatial_mean_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        mean_out,
        rsqrt_out,
        spatial_out,
        channels=CHANNELS,
        width=WIDTH,
        hw_size=hw_size,
        elements_per_channel=batch * hw_size,
        running_var_correction=running_var_correction,
        BLOCK_N=BLOCK_N,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return mean_out, rsqrt_out, spatial_out, running_mean, running_var


# db4bc2f7: [128,640,8,8] channels-last, MobileViT-S training BN/SiLU mean.
@oracle_impl(
    hardware="B200",
    point="db4bc2f7",
    CHANNELS=640,
    HEIGHT=8,
    WIDTH=8,
    BLOCK_N=128,
    BLOCK_HW=64,
    running_var_correction=1.0001220852154804,
    num_warps=8,
    num_stages=1,
)
# 86b98b1a: [128,1280,7,7] channels-last, EfficientNet-B0 training BN/SiLU mean.
@oracle_impl(
    hardware="B200",
    point="86b98b1a",
    CHANNELS=1280,
    HEIGHT=7,
    WIDTH=7,
    BLOCK_N=128,
    BLOCK_HW=64,
    running_var_correction=1.0001594642002871,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    CHANNELS: int,
    HEIGHT: int,
    WIDTH: int,
    BLOCK_N: int,
    BLOCK_HW: int,
    running_var_correction: float,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        CHANNELS=CHANNELS,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        BLOCK_N=BLOCK_N,
        BLOCK_HW=BLOCK_HW,
        running_var_correction=running_var_correction,
        num_warps=num_warps,
        num_stages=num_stages,
    )
