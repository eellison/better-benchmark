"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete LearningToPaint bf16 training-BatchNorm residual-ReLU-average-pool scope, including the bf16-to-fp32 stats input, population `var_mean(..., dim=[0,2,3], correction=0, keepdim=True)`, eps=1e-5 rsqrt, returned `[512]` invstd, saved-mean `[1,512,1,1]` view, in-place running mean/variance `copy_` updates with the captured unbiased-running-var correction, affine fp32 epilogue with the required bf16 cast before adding the bf16 residual, returned bf16 ReLU activation, and returned `[96,512]` avg-pool view, whereas Inductor schedules the BN stats/update, affine residual ReLU, and avg_pool2d/view as separate generic regions with full activation traffic; Inductor cannot fuse this today because its normalization scheduler does not keep mutable running-stat side effects, saved stats, bf16 cast boundaries, residual-ReLU, and a downstream spatial-pool consumer in one full-scope training-BN plan; the fix is SCHEDULER_FUSION: add a BN-training fusion schedule that emits mean/invstd/running-stat side outputs while sinking affine residual activation and spatial-pool consumers into the same schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


RUNNING_VAR_CORRECTION = 1.0006514657980456


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
def _bn_stats_residual_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    invstd_ptr,
    relu_ptr,
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
    RUNNING_VAR_CORRECTION_C: tl.constexpr,
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

    mean_acc = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
    m2_acc = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
    weight_acc = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
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
    var = m2 / E
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))
    channel_mask = channels < C

    old_mean = tl.load(
        running_mean_ptr + channels,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    old_var = tl.load(
        running_var_ptr + channels,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(running_mean_ptr + channels, _f32_add(_f32_mul(old_mean, 0.9), _f32_mul(mean, 0.1)), mask=channel_mask)
    tl.store(
        running_var_ptr + channels,
        _f32_add(
            _f32_mul(old_var, 0.9),
            _f32_mul(_f32_mul(var, RUNNING_VAR_CORRECTION_C), 0.1),
        ),
        mask=channel_mask,
    )
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(mean_ptr + channels, mean, mask=channel_mask)

    gamma = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    beta = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    centered = _f32_sub(vals, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    scaled = _f32_mul(normalized, gamma[:, None])
    biased = _f32_add(scaled, beta[:, None])
    y = biased.to(tl.bfloat16)
    y = (y + residual).to(tl.bfloat16)
    y = tl.where(y != y, y, tl.maximum(y, 0.0))
    tl.store(relu_ptr + offsets, y, mask=mask)


@triton.jit
def _avg_pool_4x4_kernel(
    relu_ptr,
    pooled_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = out_offsets < TOTAL
    c = out_offsets % C
    n = out_offsets // C
    hw = tl.arange(0, HW)
    relu_offsets = n[:, None] * C * HW + c[:, None] * HW + hw[None, :]
    vals = tl.load(relu_ptr + relu_offsets, mask=mask[:, None], other=0.0).to(tl.float32)
    pooled = tl.sum(vals, axis=1) * 0.0625
    tl.store(pooled_ptr + out_offsets, pooled.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape, *, numel=None):
    shape = tuple(int(dim) for dim in shape)
    if -1 not in shape:
        return shape
    if numel is None:
        raise ValueError("numel is required to resolve -1 shape dimensions")
    known = 1
    unknown_idx = None
    for idx, dim in enumerate(shape):
        if dim == -1:
            unknown_idx = idx
        else:
            known *= dim
    resolved = list(shape)
    resolved[unknown_idx] = int(numel) // known
    return tuple(resolved)


# adc76233: (T([96,512,4,4], bf16), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([96,512,4,4], bf16), S([96,-1]))
@oracle_impl(
    hardware="B200",
    point="adc76233",
    BLOCK_E=2048,
    C_BLOCK=1,
    POOL_BLOCK=128,
    num_warps=8,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    C_BLOCK: int,
    POOL_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, shape0 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    e = n * hw
    pooled_shape = _as_shape(shape0, numel=n * c)

    invstd = torch.empty_strided((c,), (1,), device=arg0_1.device, dtype=torch.float32)
    relu = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    pooled = torch.empty_strided(
        pooled_shape,
        _contiguous_stride(pooled_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _bn_stats_residual_relu_kernel[(triton.cdiv(c, C_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        invstd,
        relu,
        mean,
        c,
        h,
        w,
        e,
        arg0_1.stride(0),
        arg0_1.stride(1),
        arg0_1.stride(2),
        arg0_1.stride(3),
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        RUNNING_VAR_CORRECTION_C=RUNNING_VAR_CORRECTION,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _avg_pool_4x4_kernel[(triton.cdiv(n * c, POOL_BLOCK),)](
        relu,
        pooled,
        TOTAL=n * c,
        C=c,
        HW=hw,
        BLOCK=POOL_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return invstd, relu, pooled, mean, arg1_1, arg2_1
