"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 ConvNeXtV2 exact-GELU plus GRN forward scope as the canonical three-stage norm template, including the fp32 exact-erf GELU, required bf16 activation rounding, returned per-`(N,C)` spatial L2 norms, returned per-`N` mean-plus-eps denominator, and final bf16 channels-last affine residual output, whereas Inductor already lowers this family into comparable compact reductions and an epilogue over the rounded activation; Inductor cannot materially improve this today through local scheduler fusion, scatter-reduce, cooperative split-K, algebraic-elimination, or recompute fusion because the remaining work is the required exact GELU math, spatial norm reduction, channel mean reduction, and output store; the fix is BANDWIDTH_BOUND: record this as an at-floor GRN norm-template case unless broader pointwise/reduction codegen or launch-overhead changes move both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _gelu_bf16(x):
    return (x * 0.5) * (tl.math.erf(x * 0.7071067811865476) + 1.0)


@triton.jit
def _spatial_norm_kernel(
    x_ptr,
    norm_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    hw_offsets = tl.arange(0, BLOCK_HW)[:, None]
    hw_mask = hw_offsets < HW
    offsets = n * C * HW + hw_offsets * C + c_offsets

    x = tl.load(x_ptr + offsets, mask=hw_mask, other=0.0).to(tl.float32)
    gelu_f32 = _gelu_bf16(x).to(tl.bfloat16).to(tl.float32)
    sumsq = tl.sum(gelu_f32 * gelu_f32, axis=0)

    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(norm_ptr + n * C + channels, tl.sqrt(sumsq))


@triton.jit
def _channel_mean_kernel(
    norm_ptr,
    denom_ptr,
    ratio_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    channels = tl.arange(0, BLOCK_C)
    mask = channels < C
    norms = tl.load(norm_ptr + n * C + channels, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(norms, axis=0)
    denom = total / C + 1.0e-6
    tl.store(denom_ptr + n, denom)
    tl.store(ratio_ptr + n * C + channels, norms / denom, mask=mask)


@triton.jit
def _grn_output_kernel(
    x_ptr,
    bias_ptr,
    weight_ptr,
    ratio_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = offsets % C
    n = offsets // (C * HW)
    nc = n * C + channel

    x = tl.load(x_ptr + offsets).to(tl.float32)
    gelu_f32 = _gelu_bf16(x).to(tl.bfloat16).to(tl.float32)
    ratio = tl.load(ratio_ptr + nc).to(tl.float32)
    bias = tl.load(bias_ptr + channel, eviction_policy="evict_last").to(tl.float32)
    weight = tl.load(weight_ptr + channel, eviction_policy="evict_last").to(tl.float32)

    mul3 = gelu_f32 * ratio
    addcmul = bias + weight * mul3
    out = gelu_f32 + addcmul
    tl.store(out_ptr + offsets, out.to(tl.bfloat16))


def _out_stride(shape):
    n, c, h, w = (int(dim) for dim in shape)
    return (c * h * w, 1, w * c, c)


@oracle_impl(hardware="B200", point="3c27f190", BLOCK_HW=64, BLOCK_C=64, OUT_BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="09211248", BLOCK_HW=256, BLOCK_C=32, OUT_BLOCK=1024, num_warps=8)
@oracle_impl(hardware="B200", point="e3b8317b", BLOCK_HW=1024, BLOCK_C=4, OUT_BLOCK=1024, num_warps=8)
@oracle_impl(hardware="B200", point="ec4758af", BLOCK_HW=4096, BLOCK_C=4, OUT_BLOCK=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK_HW, BLOCK_C, OUT_BLOCK, num_warps):
    arg0_1, arg1_1, arg2_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    total = arg0_1.numel()

    norm = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    denom = torch.empty_strided((n, 1, 1, 1), (1, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    ratio = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(arg0_1.shape), _out_stride(arg0_1.shape), device=arg0_1.device, dtype=torch.bfloat16)

    _spatial_norm_kernel[(n, triton.cdiv(c, BLOCK_C))](
        arg0_1,
        norm,
        C=c,
        HW=hw,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _channel_mean_kernel[(n,)](
        norm,
        denom,
        ratio,
        C=c,
        BLOCK_C=triton.next_power_of_2(c),
        num_warps=8,
        num_stages=3,
    )
    _grn_output_kernel[(triton.cdiv(total, OUT_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        ratio,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
        BLOCK=OUT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return norm, denom, out
