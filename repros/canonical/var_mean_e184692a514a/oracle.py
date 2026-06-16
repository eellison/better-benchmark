"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet bf16 training concat-BatchNorm-ReLU scope by materializing the returned cat tensor while reducing the logical channel concatenation directly from the four source tensors, updating the running mean/variance copy_ outputs, and writing the bf16 affine-ReLU activation in one channel-tiled pass, whereas Inductor currently schedules the concat, fp32 var_mean, mutable running-stat updates, bf16 activation cast, ReLU, and returned side outputs as generic producer/consumer work with avoidable intermediate traffic; Inductor cannot do this today because its scheduler does not preserve fixed multi-input channel concatenations across multi-output normalization reductions with mutable copy_ returns and a full activation consumer; the fix is SCHEDULER_FUSION: teach the BN-training scheduler to accept static channel-concat producers, emit the mean/invstd/running-stat side outputs, and fuse the bf16 affine-ReLU epilogue into the same channel-tiled reduction plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BRANCH_C = 32
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.005128205128205


@triton.jit
def _bn_train_cat_relu_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
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

    x0_channel = tl.where(in0, channel, 0)
    x1_channel = tl.where(in1, rel, 0)
    x2_channel = tl.where(in2, rel - branch_c, 0)
    x3_channel = tl.where(in3, rel - 2 * branch_c, 0)

    x0_offsets = (n_idx * c0 + x0_channel) * hw_size + hw_idx
    branch_base = n_idx * branch_c * hw_size + hw_idx
    x1_offsets = branch_base + x1_channel * hw_size
    x2_offsets = branch_base + x2_channel * hw_size
    x3_offsets = branch_base + x3_channel * hw_size

    vals = tl.load(x0_ptr + x0_offsets, mask=mask & in0, other=0.0).to(tl.float32)
    vals += tl.load(x1_ptr + x1_offsets, mask=mask & in1, other=0.0).to(tl.float32)
    vals += tl.load(x2_ptr + x2_offsets, mask=mask & in2, other=0.0).to(tl.float32)
    vals += tl.load(x3_ptr + x3_offsets, mask=mask & in3, other=0.0).to(tl.float32)

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
        old_running_var * (1.0 - momentum)
        + var * running_var_correction * momentum,
    )
    tl.store(invstd_out_ptr + channel, invstd)
    tl.store(mean_out_ptr + channel, mean)

    out_offsets = (n_idx * channels + channel) * hw_size + hw_idx
    tl.store(cat_out_ptr + out_offsets, vals, mask=mask)

    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    y = (vals - mean) * invstd
    y = y * weight + bias
    y_bf16 = y.to(tl.bfloat16)
    relu = tl.where(y_bf16 != y_bf16, y_bf16, tl.maximum(y_bf16, 0.0))
    tl.store(relu_out_ptr + out_offsets, relu, mask=mask)


def _launch(inputs, *, C0: int, H: int, W: int, BLOCK_K: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    n = int(arg0_1.shape[0])
    channels = C0 + 3 * BRANCH_C
    hw = H * W
    elements_per_channel = n * hw

    cat = torch.empty_strided(
        (n, channels, H, W),
        (channels * hw, hw, W, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    invstd = torch.empty_strided((channels,), (1,), device=arg0_1.device, dtype=torch.float32)
    relu = torch.empty_strided(
        (n, channels, H, W),
        (channels * hw, hw, W, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _bn_train_cat_relu_kernel[(channels,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        cat,
        invstd,
        relu,
        mean,
        c0=C0,
        branch_c=BRANCH_C,
        channels=channels,
        hw_size=hw,
        elements_per_channel=elements_per_channel,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return cat, invstd, relu, mean, arg4_1, arg5_1


# 44053229: (T([4,512,7,7], bf16), T([4,32,7,7], bf16), T([4,32,7,7], bf16), T([4,32,7,7], bf16), T([608], f32), T([608], f32), T([608], f32), T([608], f32))
# 3c0535dc: (T([4,256,14,14], bf16), T([4,32,14,14], bf16), T([4,32,14,14], bf16), T([4,32,14,14], bf16), T([352], f32), T([352], f32), T([352], f32), T([352], f32))
# 7bdf6797: (T([4,128,28,28], bf16), T([4,32,28,28], bf16), T([4,32,28,28], bf16), T([4,32,28,28], bf16), T([224], f32), T([224], f32), T([224], f32), T([224], f32))
# 31bfd422: (T([4,64,56,56], bf16), T([4,32,56,56], bf16), T([4,32,56,56], bf16), T([4,32,56,56], bf16), T([160], f32), T([160], f32), T([160], f32), T([160], f32))
@oracle_impl(hardware="B200", point="44053229", C0=512, H=7, W=7, BLOCK_K=256, num_warps=4)
@oracle_impl(hardware="B200", point="3c0535dc", C0=256, H=14, W=14, BLOCK_K=1024, num_warps=4)
@oracle_impl(hardware="B200", point="7bdf6797", C0=128, H=28, W=28, BLOCK_K=4096, num_warps=8)
@oracle_impl(hardware="B200", point="31bfd422", C0=64, H=56, W=56, BLOCK_K=16384, num_warps=8)
def oracle_forward(
    inputs,
    *,
    C0: int,
    H: int,
    W: int,
    BLOCK_K: int,
    num_warps: int,
):
    return _launch(inputs, C0=C0, H=H, W=W, BLOCK_K=BLOCK_K, num_warps=num_warps)
