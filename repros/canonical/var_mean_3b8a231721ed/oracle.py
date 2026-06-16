"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DCGAN bf16 training-BatchNorm leaky-ReLU scope, including fp32 population `var_mean(..., dim=[0,2,3], correction=0, keepdim=True)` over the bf16 activation, eps=1e-5 rsqrt, returned `[C]` invstd and `[1,C,1,1]` saved mean side outputs, both mutable running-stat `copy_` aliases with the captured variance-correction literal, fp32 affine math, the explicit bf16 materialization before the leaky-ReLU compare, and the final bf16 activation, whereas Inductor lowers the normalization reduction, mutable running-stat epilogues, bf16 cast boundary, and activation consumer through generic scheduled regions; Inductor cannot keep the full training-BN side-output and activation envelope resident today while preserving the visible cast and alias boundaries; the fix is SCHEDULER_FUSION: teach the training BatchNorm lowering to fuse affine activation epilogues while still emitting saved stats and running-stat side effects."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0019569471624266


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
def _training_bn_stats_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    channel = tl.program_id(0)
    e_offsets = tl.arange(0, BLOCK_E)
    active = e_offsets < E
    n = e_offsets // HW
    hw = e_offsets - n * HW
    offsets = n * C * HW + channel * HW + hw

    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean_acc = tl.zeros([BLOCK_E], tl.float32)
    m2_acc = tl.zeros([BLOCK_E], tl.float32)
    weight_acc = tl.zeros([BLOCK_E], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(active, mean_next, mean_acc)
    m2_acc = tl.where(active, m2_next, m2_acc)
    weight_acc = tl.where(active, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 0)
    var = m2 / E
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(var, 1.0019569471624266)
    new_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))
    tl.store(mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)
    tl.store(running_mean_ptr + channel, new_mean)
    tl.store(running_var_ptr + channel, new_var)


@triton.jit
def _training_bn_stats_activation_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    channel = tl.program_id(0)
    e_offsets = tl.arange(0, BLOCK_E)
    active = e_offsets < E
    n = e_offsets // HW
    hw = e_offsets - n * HW
    offsets = n * C * HW + channel * HW + hw

    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean_acc = tl.zeros([BLOCK_E], tl.float32)
    m2_acc = tl.zeros([BLOCK_E], tl.float32)
    weight_acc = tl.zeros([BLOCK_E], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(active, mean_next, mean_acc)
    m2_acc = tl.where(active, m2_next, m2_acc)
    weight_acc = tl.where(active, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 0)
    var = m2 / E
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(var, 1.0019569471624266)
    new_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))
    tl.store(mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)
    tl.store(running_mean_ptr + channel, new_mean)
    tl.store(running_var_ptr + channel, new_var)

    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    affine_bf16 = _f32_add(scaled, bias).to(tl.bfloat16)
    rounded = affine_bf16.to(tl.float32)
    leaky = _f32_mul(rounded, 0.2)
    out = tl.where(rounded > 0.0, rounded, leaky).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=active)


@triton.jit
def _training_bn_activation_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    channel = (offsets // HW) % C
    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=active, other=0.0).to(tl.float32)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    affine_bf16 = _f32_add(scaled, bias).to(tl.bfloat16)
    rounded = affine_bf16.to(tl.float32)
    leaky = _f32_mul(rounded, 0.2)
    out = tl.where(rounded > 0.0, rounded, leaky).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=active)


def _launch(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    BLOCK_E: int,
    OUT_BLOCK: int,
    DIRECT_ACTIVATION: bool,
    stats_warps: int,
    out_warps: int,
):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    hw = H * W
    e = n * hw
    total = n * C * hw
    mean = torch.empty_strided((1, C, 1, 1), (C, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)

    if DIRECT_ACTIVATION:
        _training_bn_stats_activation_kernel[(C,)](
            x,
            running_mean,
            running_var,
            weight,
            bias,
            mean,
            invstd,
            out,
            C=C,
            HW=hw,
            E=e,
            BLOCK_E=BLOCK_E,
            num_warps=stats_warps,
            num_stages=4,
        )
    else:
        _training_bn_stats_kernel[(C,)](
            x,
            running_mean,
            running_var,
            mean,
            invstd,
            C=C,
            HW=hw,
            E=e,
            BLOCK_E=BLOCK_E,
            num_warps=stats_warps,
            num_stages=4,
        )
        _training_bn_activation_kernel[(triton.cdiv(total, OUT_BLOCK),)](
            x,
            weight,
            bias,
            mean,
            invstd,
            out,
            TOTAL=total,
            C=C,
            HW=hw,
            BLOCK=OUT_BLOCK,
            num_warps=out_warps,
            num_stages=4,
        )

    return invstd, out, mean, running_mean, running_var


# 1918a7da: (T([32,512,4,4], bf16), T([512], f32), T([512], f32), ...)
@oracle_impl(hardware="B200", point="1918a7da", C=512, H=4, W=4, BLOCK_E=512, OUT_BLOCK=256, DIRECT_ACTIVATION=True, stats_warps=8, out_warps=4)
# 1e748ae3: (T([32,256,8,8], bf16), T([256], f32), T([256], f32), ...)
@oracle_impl(hardware="B200", point="1e748ae3", C=256, H=8, W=8, BLOCK_E=2048, OUT_BLOCK=256, DIRECT_ACTIVATION=True, stats_warps=8, out_warps=4)
# c42a1f04: (T([32,128,16,16], bf16), T([128], f32), T([128], f32), ...)
@oracle_impl(hardware="B200", point="c42a1f04", C=128, H=16, W=16, BLOCK_E=8192, OUT_BLOCK=256, DIRECT_ACTIVATION=True, stats_warps=8, out_warps=4)
def oracle_forward(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    BLOCK_E: int,
    OUT_BLOCK: int,
    DIRECT_ACTIVATION: bool,
    stats_warps: int,
    out_warps: int,
):
    return _launch(
        inputs,
        C=C,
        H=H,
        W=W,
        BLOCK_E=BLOCK_E,
        OUT_BLOCK=OUT_BLOCK,
        DIRECT_ACTIVATION=DIRECT_ACTIVATION,
        stats_warps=stats_warps,
        out_warps=out_warps,
    )
