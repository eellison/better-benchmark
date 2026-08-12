"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 ConvNeXtV2 exact-GELU plus Global Response Normalization forward scope as a channels-last norm template: reduce the bf16-rounded exact-GELU activation squared over spatial dimensions into bf16 per-(N,C) L2 norms, reduce those bf16 norms across channels into the bf16 per-sample mean, and apply every bf16 GRN epilogue rounding boundary before the channels-last output store, recomputing GELU for smaller points and materializing it with contiguous stores for the largest point; Inductor already emits comparable compact reductions and output epilogues for this required norm template, so there is no confirmed local scheduler-fusion, scatter-reduce, cooperative split-K, algebraic-elimination, or recompute-fusion opportunity left in this measured scope; the fix is BANDWIDTH_BOUND: record this as an at-floor GRN norm-template case unless broader pointwise/reduction codegen or launch-overhead changes move both implementations."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gelu_bf16(x):
    gelu = (x * 0.5) * (libdevice.erf(x * 0.7071067811865476) + 1.0)
    return gelu.to(tl.bfloat16)


@triton.jit
def _gelu_store_bf16_kernel(
    x_ptr,
    gelu_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(gelu_ptr + offsets, _gelu_bf16(x), mask=mask)


@triton.jit
def _spatial_norm_recompute_bf16_kernel(
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
    mask = (c_offsets < C) & (hw_offsets < HW)
    offsets = n * C * HW + hw_offsets * C + c_offsets

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gelu = _gelu_bf16(x).to(tl.float32)
    sumsq = tl.sum(tl.where(mask, gelu * gelu, 0.0), axis=0)

    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    norm = libdevice.sqrt(sumsq).to(tl.bfloat16)
    tl.store(norm_ptr + n * C + channels, norm, mask=channels < C)


@triton.jit
def _spatial_norm_from_gelu_bf16_kernel(
    gelu_ptr,
    norm_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    hw_offsets = tl.arange(0, BLOCK_HW)[:, None]
    mask = (c_offsets < C) & (hw_offsets < HW)
    offsets = n * C * HW + hw_offsets * C + c_offsets

    gelu = tl.load(gelu_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(tl.where(mask, gelu * gelu, 0.0), axis=0)

    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    norm = libdevice.sqrt(sumsq).to(tl.bfloat16)
    tl.store(norm_ptr + n * C + channels, norm, mask=channels < C)


@triton.jit
def _channel_mean_bf16_kernel(
    norm_ptr,
    mean_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    channels = tl.arange(0, BLOCK_C)
    mask = channels < C
    norms = tl.load(norm_ptr + n * C + channels, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(mask, norms, 0.0), axis=0)
    mean = (total / C).to(tl.bfloat16)
    tl.store(mean_ptr + n, mean)


@triton.jit
def _grn_output_recompute_bf16_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    norm_ptr,
    mean_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    channel = offsets % C
    n = offsets // (C * HW)
    nc = n * C + channel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gelu = _gelu_bf16(x).to(tl.float32)
    norm = tl.load(norm_ptr + nc, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + n, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    denom = (mean + 1.0e-6).to(tl.bfloat16).to(tl.float32)
    ratio = (norm / denom).to(tl.bfloat16).to(tl.float32)
    mul3 = (gelu * ratio).to(tl.bfloat16).to(tl.float32)
    addcmul = (bias + weight * mul3).to(tl.bfloat16).to(tl.float32)
    out = (gelu + addcmul).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _grn_output_stored_bf16_kernel(
    bias_ptr,
    weight_ptr,
    gelu_ptr,
    norm_ptr,
    mean_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    channel = offsets % C
    n = offsets // (C * HW)
    nc = n * C + channel

    gelu = tl.load(gelu_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + nc, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + n, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    denom = (mean + 1.0e-6).to(tl.bfloat16).to(tl.float32)
    ratio = (norm / denom).to(tl.bfloat16).to(tl.float32)
    mul3 = (gelu * ratio).to(tl.bfloat16).to(tl.float32)
    addcmul = (bias + weight * mul3).to(tl.bfloat16).to(tl.float32)
    out = (gelu + addcmul).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="428ddc1f", BLOCK_HW=128, BLOCK_C=32, STORE_GELU=True, num_warps=4)
@oracle_impl(hardware="B200", point="4a0c78a6", BLOCK_HW=512, BLOCK_C=16, STORE_GELU=False, num_warps=8)
@oracle_impl(hardware="B200", point="7b1e363f", BLOCK_HW=2048, BLOCK_C=4, STORE_GELU=False, num_warps=8)
@oracle_impl(hardware="B200", point="78b1c15c", BLOCK_HW=8192, BLOCK_C=8, STORE_GELU=True, num_warps=8)
def oracle_forward(inputs, *, BLOCK_HW, BLOCK_C, STORE_GELU, num_warps):
    bias, weight, x = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    total = x.numel()

    norm = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.bfloat16)
    mean = torch.empty_strided((n, 1, 1, 1), (1, 1, 1, 1), device=x.device, dtype=torch.bfloat16)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)

    if STORE_GELU:
        gelu = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
        _gelu_store_bf16_kernel[(triton.cdiv(total, 1024),)](
            x,
            gelu,
            TOTAL=total,
            BLOCK=1024,
            num_warps=4,
            num_stages=3,
        )
        _spatial_norm_from_gelu_bf16_kernel[(n, triton.cdiv(c, BLOCK_C))](
            gelu,
            norm,
            C=c,
            HW=hw,
            BLOCK_HW=BLOCK_HW,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        _spatial_norm_recompute_bf16_kernel[(n, triton.cdiv(c, BLOCK_C))](
            x,
            norm,
            C=c,
            HW=hw,
            BLOCK_HW=BLOCK_HW,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=3,
        )

    _channel_mean_bf16_kernel[(n,)](
        norm,
        mean,
        C=c,
        BLOCK_C=triton.next_power_of_2(c),
        num_warps=8,
        num_stages=3,
    )
    if STORE_GELU:
        _grn_output_stored_bf16_kernel[(triton.cdiv(total, 1024),)](
            bias,
            weight,
            gelu,
            norm,
            mean,
            out,
            TOTAL=total,
            C=c,
            HW=hw,
            BLOCK=1024,
            num_warps=4,
            num_stages=3,
        )
    else:
        _grn_output_recompute_bf16_kernel[(triton.cdiv(total, 1024),)](
            bias,
            weight,
            x,
            norm,
            mean,
            out,
            TOTAL=total,
            C=c,
            HW=hw,
            BLOCK=1024,
            num_warps=4,
            num_stages=3,
        )
    return out
