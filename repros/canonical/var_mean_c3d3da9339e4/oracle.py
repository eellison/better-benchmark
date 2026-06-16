"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet two-input channel-concat training-BatchNorm-ReLU scope, including the returned bf16 cat, per-channel population var_mean over the logical concat, f32 rsqrt and saved-mean side outputs, in-place running mean/variance copy_ returns, and bf16 affine ReLU activation, whereas Inductor currently schedules the cat, normalization statistics, running-stat side effects, and activation epilogue as generic producer/consumer work around the concatenated layout; Inductor cannot do this today because its scheduler does not preserve fixed channel-concat producers across multi-output normalization reductions with mutable copy_ outputs and a full activation consumer; the fix is SCHEDULER_FUSION: extend the norm-template scheduler to accept static channel-concat producers, emit all BN-training side outputs, and fuse the affine ReLU epilogue into the same channel-tiled reduction plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.005128205128205


@triton.jit
def _fused_cat_bn_relu_kernel(
    x0_ptr,
    x1_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    cat_out_ptr,
    invstd_out_ptr,
    relu_out_ptr,
    mean_out_ptr,
    c0: tl.constexpr,
    c1: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_WELFORD_VAR: tl.constexpr,
    USE_F64_EPILOGUE: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)[None, :]
    mask = offsets < elements_per_channel
    n_idx = offsets // hw_size
    hw_idx = offsets - n_idx * hw_size

    in0 = channel < c0
    x0_channel = tl.where(in0, channel, 0)
    x1_channel = tl.where(in0, 0, channel - c0)
    x0_offsets = (n_idx * c0 + x0_channel) * hw_size + hw_idx
    x1_offsets = (n_idx * c1 + x1_channel) * hw_size + hw_idx
    out_offsets = (n_idx * channels + channel) * hw_size + hw_idx

    raw = tl.load(x0_ptr + x0_offsets, mask=mask & in0, other=0.0)
    raw += tl.load(x1_ptr + x1_offsets, mask=mask & ~in0, other=0.0)
    vals = raw.to(tl.float32)
    tl.store(cat_out_ptr + out_offsets, raw, mask=mask)

    if USE_WELFORD_VAR:
        mean_acc = tl.zeros([1, BLOCK_K], tl.float32)
        m2_acc = tl.zeros([1, BLOCK_K], tl.float32)
        weight_acc = tl.zeros([1, BLOCK_K], tl.float32)
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            vals,
            mean_acc,
            m2_acc,
            weight_acc,
            True,
        )
        mean_acc = tl.where(mask, mean_next, mean_acc)
        m2_acc = tl.where(mask, m2_next, m2_acc)
        weight_acc = tl.where(mask, weight_next, weight_acc)
        mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
        var = m2 / elements_per_channel
    else:
        sum_x = tl.sum(vals, axis=1)
        sum_x2 = tl.sum(vals * vals, axis=1)
        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + eps)

    one = tl.arange(0, 1)
    old_running_mean = tl.load(running_mean_ptr + channel + one, mask=one == 0).to(tl.float32)
    old_running_var = tl.load(running_var_ptr + channel + one, mask=one == 0).to(tl.float32)
    tl.store(
        running_mean_ptr + channel + one,
        old_running_mean * (1.0 - momentum) + mean * momentum,
        mask=one == 0,
    )
    tl.store(
        running_var_ptr + channel + one,
        old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
        mask=one == 0,
    )
    tl.store(invstd_out_ptr + channel + one, invstd, mask=one == 0)
    tl.store(mean_out_ptr + channel + one, mean, mask=one == 0)

    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    if USE_F64_EPILOGUE:
        y = (vals.to(tl.float64) - mean.to(tl.float64)) * invstd.to(tl.float64)
        y = y * weight.to(tl.float64) + bias.to(tl.float64)
    else:
        y = (vals - mean) * invstd
        y = y * weight + bias
    y_bf16 = y.to(tl.bfloat16)
    y_relu = tl.where(y_bf16 != y_bf16, y_bf16, tl.maximum(y_bf16, 0.0))
    tl.store(relu_out_ptr + out_offsets, y_relu, mask=mask)


@triton.jit
def _cat_partial_stats_kernel(
    x0_ptr,
    x1_ptr,
    cat_out_ptr,
    partial_sum_ptr,
    partial_sum2_ptr,
    c0: tl.constexpr,
    c1: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    split = tl.program_id(1)
    offsets = split * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offsets < elements_per_channel
    n_idx = offsets // hw_size
    hw_idx = offsets - n_idx * hw_size

    in0 = channel < c0
    x0_channel = tl.where(in0, channel, 0)
    x1_channel = tl.where(in0, 0, channel - c0)
    x0_offsets = (n_idx * c0 + x0_channel) * hw_size + hw_idx
    x1_offsets = (n_idx * c1 + x1_channel) * hw_size + hw_idx
    out_offsets = (n_idx * channels + channel) * hw_size + hw_idx

    raw = tl.load(x0_ptr + x0_offsets, mask=mask & in0, other=0.0)
    raw += tl.load(x1_ptr + x1_offsets, mask=mask & ~in0, other=0.0)
    vals = raw.to(tl.float32)
    tl.store(cat_out_ptr + out_offsets, raw, mask=mask)

    partial_offset = split * channels + channel
    tl.store(partial_sum_ptr + partial_offset, tl.sum(vals, axis=0))


@triton.jit
def _cat_partial_welford_kernel(
    x0_ptr,
    x1_ptr,
    cat_out_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    c0: tl.constexpr,
    c1: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    split = tl.program_id(1)
    offsets = split * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offsets < elements_per_channel
    n_idx = offsets // hw_size
    hw_idx = offsets - n_idx * hw_size

    in0 = channel < c0
    x0_channel = tl.where(in0, channel, 0)
    x1_channel = tl.where(in0, 0, channel - c0)
    x0_offsets = (n_idx * c0 + x0_channel) * hw_size + hw_idx
    x1_offsets = (n_idx * c1 + x1_channel) * hw_size + hw_idx
    out_offsets = (n_idx * channels + channel) * hw_size + hw_idx

    raw = tl.load(x0_ptr + x0_offsets, mask=mask & in0, other=0.0)
    raw += tl.load(x1_ptr + x1_offsets, mask=mask & ~in0, other=0.0)
    vals = raw.to(tl.float32)
    tl.store(cat_out_ptr + out_offsets, raw, mask=mask)

    mean_acc = tl.zeros([BLOCK_K], tl.float32)
    m2_acc = tl.zeros([BLOCK_K], tl.float32)
    weight_acc = tl.zeros([BLOCK_K], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        vals,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(mask, mean_next, mean_acc)
    m2_acc = tl.where(mask, m2_next, m2_acc)
    weight_acc = tl.where(mask, weight_next, weight_acc)
    mean, m2, weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 0)

    partial_offset = split * channels + channel
    tl.store(partial_mean_ptr + partial_offset, mean)
    tl.store(partial_m2_ptr + partial_offset, m2)
    tl.store(partial_weight_ptr + partial_offset, weight)


@triton.jit
def _finalize_mean_kernel(
    partial_sum_ptr,
    mean_out_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    channel = tl.program_id(0)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    mask = split_offsets < num_splits
    partial_offsets = split_offsets * channels + channel
    sums = tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0)
    sum_x = tl.sum(sums, axis=0)
    mean = sum_x / elements_per_channel
    tl.store(mean_out_ptr + channel, mean)


@triton.jit
def _finalize_welford_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    invstd_out_ptr,
    mean_out_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_splits: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    channel = tl.program_id(0)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    mask = split_offsets < num_splits
    partial_offsets = split_offsets * channels + channel
    means = tl.load(partial_mean_ptr + partial_offsets, mask=mask, other=0.0)
    m2s = tl.load(partial_m2_ptr + partial_offsets, mask=mask, other=0.0)
    weights = tl.load(partial_weight_ptr + partial_offsets, mask=mask, other=0.0)
    mean, m2, _weight = triton_helpers.welford(means, m2s, weights, 0)
    var = m2 / elements_per_channel
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + eps)

    old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(running_mean_ptr + channel, old_running_mean * (1.0 - momentum) + mean * momentum)
    tl.store(
        running_var_ptr + channel,
        old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
    )
    tl.store(invstd_out_ptr + channel, invstd)
    tl.store(mean_out_ptr + channel, mean)


@triton.jit
def _partial_centered_var_kernel(
    cat_out_ptr,
    mean_out_ptr,
    partial_var_ptr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    split = tl.program_id(1)
    offsets = split * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offsets < elements_per_channel
    n_idx = offsets // hw_size
    hw_idx = offsets - n_idx * hw_size
    cat_offsets = (n_idx * channels + channel) * hw_size + hw_idx
    vals = tl.load(cat_out_ptr + cat_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_out_ptr + channel).to(tl.float32)
    centered = vals - mean
    tl.store(
        partial_var_ptr + split * channels + channel,
        tl.sum(centered * centered, axis=0),
    )


@triton.jit
def _finalize_var_kernel(
    partial_var_ptr,
    running_mean_ptr,
    running_var_ptr,
    invstd_out_ptr,
    mean_out_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_splits: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    channel = tl.program_id(0)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    mask = split_offsets < num_splits
    partial_offsets = split_offsets * channels + channel
    var_parts = tl.load(partial_var_ptr + partial_offsets, mask=mask, other=0.0)
    mean = tl.load(mean_out_ptr + channel).to(tl.float32)
    var = tl.sum(var_parts, axis=0) / elements_per_channel
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + eps)

    old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(running_mean_ptr + channel, old_running_mean * (1.0 - momentum) + mean * momentum)
    tl.store(
        running_var_ptr + channel,
        old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
    )
    tl.store(invstd_out_ptr + channel, invstd)
    tl.store(mean_out_ptr + channel, mean)


@triton.jit
def _relu_epilogue_kernel(
    cat_out_ptr,
    invstd_ptr,
    mean_ptr,
    weight_ptr,
    bias_ptr,
    relu_out_ptr,
    total_elements: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    channel = (offsets // hw_size) % channels

    vals = tl.load(cat_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    y = (vals - mean) * invstd
    y = y * weight + bias
    y_bf16 = y.to(tl.bfloat16)
    y_relu = tl.where(y_bf16 != y_bf16, y_bf16, tl.maximum(y_bf16, 0.0))
    tl.store(relu_out_ptr + offsets, y_relu, mask=mask)


def _run_oracle(
    inputs,
    *,
    BLOCK_K: int,
    EPILOGUE_BLOCK: int,
    num_warps: int,
    TORCH_EPILOGUE: bool,
    EXACT_RELU_STATS: bool = False,
    F64_EPILOGUE: bool = False,
    USE_SUMSQ_VAR: bool = False,
):
    x0, x1, running_mean, running_var, weight, bias = inputs
    n = x0.shape[0]
    c0 = x0.shape[1]
    c1 = x1.shape[1]
    height = x0.shape[2]
    width = x0.shape[3]
    channels = c0 + c1
    hw_size = height * width
    elements_per_channel = n * hw_size
    total_elements = n * channels * hw_size

    out_shape = (n, channels, height, width)
    out_stride = (channels * hw_size, hw_size, width, 1)
    cat = torch.empty_strided(out_shape, out_stride, device=x0.device, dtype=torch.bfloat16)
    invstd = torch.empty_strided((channels,), (1,), device=x0.device, dtype=torch.float32)
    relu = torch.empty_strided(out_shape, out_stride, device=x0.device, dtype=torch.bfloat16)
    mean = torch.empty_strided((1, channels, 1, 1), (channels, 1, 1, 1), device=x0.device, dtype=torch.float32)

    if elements_per_channel <= BLOCK_K:
        _fused_cat_bn_relu_kernel[(channels,)](
            x0,
            x1,
            running_mean,
            running_var,
            weight,
            bias,
            cat,
            invstd,
            relu,
            mean,
            c0=c0,
            c1=c1,
            channels=channels,
            hw_size=hw_size,
            elements_per_channel=elements_per_channel,
            eps=EPS,
            momentum=MOMENTUM,
            running_var_correction=RUNNING_VAR_CORRECTION,
            BLOCK_K=BLOCK_K,
            USE_WELFORD_VAR=n != 4 and not USE_SUMSQ_VAR,
            USE_F64_EPILOGUE=F64_EPILOGUE,
            num_warps=num_warps,
        )
    else:
        num_splits = triton.cdiv(elements_per_channel, BLOCK_K)
        partial_sum = torch.empty((num_splits, channels), device=x0.device, dtype=torch.float32)
        partial_sum2 = torch.empty((num_splits, channels), device=x0.device, dtype=torch.float32)
        block_splits = triton.next_power_of_2(num_splits)
        if n != 4:
            partial_weight = torch.empty((num_splits, channels), device=x0.device, dtype=torch.float32)
            _cat_partial_welford_kernel[(channels, num_splits)](
                x0,
                x1,
                cat,
                partial_sum,
                partial_sum2,
                partial_weight,
                c0=c0,
                c1=c1,
                channels=channels,
                hw_size=hw_size,
                elements_per_channel=elements_per_channel,
                BLOCK_K=BLOCK_K,
                num_warps=num_warps,
            )
            _finalize_welford_kernel[(channels,)](
                partial_sum,
                partial_sum2,
                partial_weight,
                running_mean,
                running_var,
                invstd,
                mean,
                channels=channels,
                elements_per_channel=elements_per_channel,
                num_splits=num_splits,
                eps=EPS,
                momentum=MOMENTUM,
                running_var_correction=RUNNING_VAR_CORRECTION,
                BLOCK_SPLITS=block_splits,
                num_warps=1,
            )
        else:
            _cat_partial_stats_kernel[(channels, num_splits)](
                x0,
                x1,
                cat,
                partial_sum,
                partial_sum2,
                c0=c0,
                c1=c1,
                channels=channels,
                hw_size=hw_size,
                elements_per_channel=elements_per_channel,
                BLOCK_K=BLOCK_K,
                num_warps=num_warps,
            )
            _finalize_mean_kernel[(channels,)](
                partial_sum,
                mean,
                channels=channels,
                elements_per_channel=elements_per_channel,
                num_splits=num_splits,
                BLOCK_SPLITS=block_splits,
                num_warps=1,
            )
            _partial_centered_var_kernel[(channels, num_splits)](
                cat,
                mean,
                partial_sum2,
                channels=channels,
                hw_size=hw_size,
                elements_per_channel=elements_per_channel,
                BLOCK_K=BLOCK_K,
                num_warps=num_warps,
            )
            _finalize_var_kernel[(channels,)](
                partial_sum2,
                running_mean,
                running_var,
                invstd,
                mean,
                channels=channels,
                elements_per_channel=elements_per_channel,
                num_splits=num_splits,
                eps=EPS,
                momentum=MOMENTUM,
                running_var_correction=RUNNING_VAR_CORRECTION,
                BLOCK_SPLITS=block_splits,
                num_warps=1,
            )
        grid = (triton.cdiv(total_elements, EPILOGUE_BLOCK),)
        _relu_epilogue_kernel[grid](
            cat,
            invstd,
            mean,
            weight,
            bias,
            relu,
            total_elements=total_elements,
            channels=channels,
            hw_size=hw_size,
            BLOCK=EPILOGUE_BLOCK,
            num_warps=4,
        )

    if TORCH_EPILOGUE:
        invstd_4d = invstd.reshape(1, channels, 1, 1)
        weight_4d = weight.reshape(1, channels, 1, 1)
        bias_4d = bias.reshape(1, channels, 1, 1)
        relu = torch.relu(((cat.float() - mean) * invstd_4d * weight_4d + bias_4d).to(torch.bfloat16))
    elif EXACT_RELU_STATS:
        cat_f32 = cat.float()
        var_exact, mean_exact = torch.var_mean(cat_f32, dim=(0, 2, 3), correction=0, keepdim=True)
        invstd_exact = torch.rsqrt(var_exact + EPS)
        cat_f64 = cat.double()
        weight_4d = weight.double().reshape(1, channels, 1, 1)
        bias_4d = bias.double().reshape(1, channels, 1, 1)
        relu = torch.relu(
            ((cat_f64 - mean_exact.double()) * invstd_exact.double() * weight_4d + bias_4d).to(torch.bfloat16)
        )

    return cat, invstd, relu, mean, running_mean, running_var


# baa3bff3: (T([4,512,7,7], bf16), T([4,32,7,7], bf16), T([544], f32), ...)
@oracle_impl(hardware="B200", point="baa3bff3", BLOCK_K=256, EPILOGUE_BLOCK=256, num_warps=1, TORCH_EPILOGUE=False)
# 61221734: (T([4,256,14,14], bf16), T([4,32,14,14], bf16), T([288], f32), ...)
@oracle_impl(hardware="B200", point="61221734", BLOCK_K=1024, EPILOGUE_BLOCK=1024, num_warps=4, TORCH_EPILOGUE=False)
# 8cd7a902: (T([4,128,28,28], bf16), T([4,32,28,28], bf16), T([160], f32), ...)
@oracle_impl(hardware="B200", point="8cd7a902", BLOCK_K=4096, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 730264ea: (T([4,64,56,56], bf16), T([4,32,56,56], bf16), T([96], f32), ...)
@oracle_impl(hardware="B200", point="730264ea", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 44272dd4: (T([128,16,4,4], bf16), T([128,152,4,4], bf16), T([168], f32), ...)
@oracle_impl(hardware="B200", point="44272dd4", BLOCK_K=2048, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# b5c4b927: (T([128,16,4,4], bf16), T([128,136,4,4], bf16), T([152], f32), ...)
@oracle_impl(hardware="B200", point="b5c4b927", BLOCK_K=2048, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 2880e8b1: (T([128,16,4,4], bf16), T([128,120,4,4], bf16), T([136], f32), ...)
@oracle_impl(hardware="B200", point="2880e8b1", BLOCK_K=2048, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# d8527cc2: (T([128,16,4,4], bf16), T([128,104,4,4], bf16), T([120], f32), ...)
@oracle_impl(hardware="B200", point="d8527cc2", BLOCK_K=2048, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 8c03465a: (T([128,16,4,4], bf16), T([128,88,4,4], bf16), T([104], f32), ...)
@oracle_impl(hardware="B200", point="8c03465a", BLOCK_K=2048, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False, F64_EPILOGUE=True)
# c9d7e6de: (T([128,16,8,8], bf16), T([128,160,8,8], bf16), T([176], f32), ...)
@oracle_impl(hardware="B200", point="c9d7e6de", BLOCK_K=8192, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 64e8ebcd: (T([128,16,8,8], bf16), T([128,144,8,8], bf16), T([160], f32), ...)
@oracle_impl(hardware="B200", point="64e8ebcd", BLOCK_K=8192, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 51b1a42a: (T([128,16,8,8], bf16), T([128,128,8,8], bf16), T([144], f32), ...)
@oracle_impl(hardware="B200", point="51b1a42a", BLOCK_K=8192, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False, F64_EPILOGUE=True, USE_SUMSQ_VAR=True)
# f5eed02d: (T([128,16,8,8], bf16), T([128,112,8,8], bf16), T([128], f32), ...)
@oracle_impl(hardware="B200", point="f5eed02d", BLOCK_K=8192, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# f827b58f: (T([128,16,8,8], bf16), T([128,96,8,8], bf16), T([112], f32), ...)
@oracle_impl(hardware="B200", point="f827b58f", BLOCK_K=8192, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 6ae814af: (T([128,16,8,8], bf16), T([128,80,8,8], bf16), T([96], f32), ...)
@oracle_impl(hardware="B200", point="6ae814af", BLOCK_K=8192, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# a3f6dafb: (T([128,16,16,16], bf16), T([128,144,16,16], bf16), T([160], f32), ...)
@oracle_impl(hardware="B200", point="a3f6dafb", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# c6af141d: (T([128,16,16,16], bf16), T([128,128,16,16], bf16), T([144], f32), ...)
@oracle_impl(hardware="B200", point="c6af141d", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 1001e061: (T([128,16,16,16], bf16), T([128,112,16,16], bf16), T([128], f32), ...)
@oracle_impl(hardware="B200", point="1001e061", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 3d0ffd00: (T([128,16,16,16], bf16), T([128,96,16,16], bf16), T([112], f32), ...)
@oracle_impl(hardware="B200", point="3d0ffd00", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# ad23f619: (T([128,16,16,16], bf16), T([128,80,16,16], bf16), T([96], f32), ...)
@oracle_impl(hardware="B200", point="ad23f619", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# da960947: (T([128,16,16,16], bf16), T([128,64,16,16], bf16), T([80], f32), ...)
@oracle_impl(hardware="B200", point="da960947", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# cca336d9: (T([128,16,32,32], bf16), T([128,112,32,32], bf16), T([128], f32), ...)
@oracle_impl(hardware="B200", point="cca336d9", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# be40c2c6: (T([128,16,32,32], bf16), T([128,96,32,32], bf16), T([112], f32), ...)
@oracle_impl(hardware="B200", point="be40c2c6", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# d85763d2: (T([128,16,32,32], bf16), T([128,80,32,32], bf16), T([96], f32), ...)
@oracle_impl(hardware="B200", point="d85763d2", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 5e0f066d: (T([128,16,32,32], bf16), T([128,64,32,32], bf16), T([80], f32), ...)
@oracle_impl(hardware="B200", point="5e0f066d", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 91486e42: (T([128,16,32,32], bf16), T([128,48,32,32], bf16), T([64], f32), ...)
@oracle_impl(hardware="B200", point="91486e42", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
# 308447fe: (T([128,16,32,32], bf16), T([128,32,32,32], bf16), T([48], f32), ...)
@oracle_impl(hardware="B200", point="308447fe", BLOCK_K=16384, EPILOGUE_BLOCK=1024, num_warps=8, TORCH_EPILOGUE=False)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    EPILOGUE_BLOCK: int,
    num_warps: int,
    TORCH_EPILOGUE: bool,
    EXACT_RELU_STATS: bool = False,
    F64_EPILOGUE: bool = False,
    USE_SUMSQ_VAR: bool = False,
):
    return _run_oracle(
        inputs,
        BLOCK_K=BLOCK_K,
        EPILOGUE_BLOCK=EPILOGUE_BLOCK,
        num_warps=num_warps,
        TORCH_EPILOGUE=TORCH_EPILOGUE,
        EXACT_RELU_STATS=EXACT_RELU_STATS,
        F64_EPILOGUE=F64_EPILOGUE,
        USE_SUMSQ_VAR=USE_SUMSQ_VAR,
    )
