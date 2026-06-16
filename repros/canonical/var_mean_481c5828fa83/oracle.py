"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet five-input channel-concat training-BatchNorm-ReLU scope, including the returned bf16 cat, per-channel population var_mean over the logical concat, f32 rsqrt and saved-mean side outputs, in-place running mean/variance copy_ returns, and bf16 affine ReLU activation, whereas Inductor currently schedules the cat, normalization statistics, running-stat side effects, and activation epilogue as generic producer/consumer work around the concatenated layout; Inductor cannot do this today because its scheduler does not preserve fixed multi-input channel-concat producers across multi-output normalization reductions with mutable copy_ outputs and a full activation consumer; the fix is SCHEDULER_FUSION: extend the norm-template scheduler to accept static multi-input channel-concat producers, emit all BN-training side outputs, and fuse the affine ReLU epilogue into the same channel-tiled reduction plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.005128205128205
BRANCHES = 4


@triton.jit
def _fused_cat_bn_relu_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    x4_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    cat_out_ptr,
    invstd_out_ptr,
    relu_out_ptr,
    mean_out_ptr,
    c0: tl.constexpr,
    branch_c: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_WELFORD_VAR: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < elements_per_channel
    n_idx = offsets // hw_size
    hw_idx = offsets - n_idx * hw_size

    rel = channel - c0
    in0 = channel < c0
    in1 = (rel >= 0) & (rel < branch_c)
    in2 = (rel >= branch_c) & (rel < 2 * branch_c)
    in3 = (rel >= 2 * branch_c) & (rel < 3 * branch_c)
    in4 = (rel >= 3 * branch_c) & (rel < 4 * branch_c)

    x0_channel = tl.where(in0, channel, 0)
    x1_channel = tl.where(in1, rel, 0)
    x2_channel = tl.where(in2, rel - branch_c, 0)
    x3_channel = tl.where(in3, rel - 2 * branch_c, 0)
    x4_channel = tl.where(in4, rel - 3 * branch_c, 0)

    x0_offsets = (n_idx * c0 + x0_channel) * hw_size + hw_idx
    branch_base = n_idx * branch_c * hw_size + hw_idx
    x1_offsets = branch_base + x1_channel * hw_size
    x2_offsets = branch_base + x2_channel * hw_size
    x3_offsets = branch_base + x3_channel * hw_size
    x4_offsets = branch_base + x4_channel * hw_size
    out_offsets = (n_idx * channels + channel) * hw_size + hw_idx

    raw = tl.load(x0_ptr + x0_offsets, mask=mask & in0, other=0.0)
    raw += tl.load(x1_ptr + x1_offsets, mask=mask & in1, other=0.0)
    raw += tl.load(x2_ptr + x2_offsets, mask=mask & in2, other=0.0)
    raw += tl.load(x3_ptr + x3_offsets, mask=mask & in3, other=0.0)
    raw += tl.load(x4_ptr + x4_offsets, mask=mask & in4, other=0.0)
    vals = raw.to(tl.float32)
    tl.store(cat_out_ptr + out_offsets, raw, mask=mask)

    if USE_WELFORD_VAR:
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
        mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 0)
        var = m2 / elements_per_channel
    else:
        sum_x = tl.sum(vals, axis=0)
        sum_x2 = tl.sum(vals * vals, axis=0)
        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
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

    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    y = (vals - mean) * invstd
    y = y * weight + bias
    y_bf16 = y.to(tl.bfloat16)
    y_relu = tl.where(y_bf16 != y_bf16, y_bf16, tl.maximum(y_bf16, 0.0))
    tl.store(relu_out_ptr + out_offsets, y_relu, mask=mask)


def _run_oracle(inputs, *, BLOCK_K: int, num_warps: int, USE_WELFORD_VAR: bool):
    x0, x1, x2, x3, x4, running_mean, running_var, weight, bias = inputs
    n = x0.shape[0]
    c0 = x0.shape[1]
    branch_c = x1.shape[1]
    height = x0.shape[2]
    width = x0.shape[3]
    channels = c0 + BRANCHES * branch_c
    hw_size = height * width
    elements_per_channel = n * hw_size

    out_shape = (n, channels, height, width)
    out_stride = (channels * hw_size, hw_size, width, 1)
    cat = torch.empty_strided(out_shape, out_stride, device=x0.device, dtype=torch.bfloat16)
    invstd = torch.empty_strided((channels,), (1,), device=x0.device, dtype=torch.float32)
    relu = torch.empty_strided(out_shape, out_stride, device=x0.device, dtype=torch.bfloat16)
    mean = torch.empty_strided((1, channels, 1, 1), (channels, 1, 1, 1), device=x0.device, dtype=torch.float32)

    _fused_cat_bn_relu_kernel[(channels,)](
        x0,
        x1,
        x2,
        x3,
        x4,
        running_mean,
        running_var,
        weight,
        bias,
        cat,
        invstd,
        relu,
        mean,
        c0=c0,
        branch_c=branch_c,
        channels=channels,
        hw_size=hw_size,
        elements_per_channel=elements_per_channel,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_K=BLOCK_K,
        USE_WELFORD_VAR=USE_WELFORD_VAR,
        num_warps=num_warps,
    )
    return cat, invstd, relu, mean, running_mean, running_var


# 85964560: (T([4,512,7,7], bf16), four T([4,32,7,7], bf16), T([640], f32), ...)
@oracle_impl(hardware="B200", point="85964560", BLOCK_K=256, num_warps=1, USE_WELFORD_VAR=False)
# 4d94142f: (T([4,256,14,14], bf16), four T([4,32,14,14], bf16), T([384], f32), ...)
@oracle_impl(hardware="B200", point="4d94142f", BLOCK_K=1024, num_warps=4, USE_WELFORD_VAR=False)
# e0b577eb: (T([4,128,28,28], bf16), four T([4,32,28,28], bf16), T([256], f32), ...)
@oracle_impl(hardware="B200", point="e0b577eb", BLOCK_K=4096, num_warps=8, USE_WELFORD_VAR=False)
# 920aacff: (T([4,64,56,56], bf16), four T([4,32,56,56], bf16), T([192], f32), ...)
@oracle_impl(hardware="B200", point="920aacff", BLOCK_K=16384, num_warps=8, USE_WELFORD_VAR=True)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int, USE_WELFORD_VAR: bool):
    return _run_oracle(inputs, BLOCK_K=BLOCK_K, num_warps=num_warps, USE_WELFORD_VAR=USE_WELFORD_VAR)
