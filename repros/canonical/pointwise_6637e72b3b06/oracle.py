"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileNetV2 inference batchnorm-affine residual epilogue as one shape-specialized Triton storage-order pointwise kernel, preserving the captured f32 sqrt/reciprocal normalization, f32 affine math, explicit bf16 affine cast before the residual add, and final bf16 output scope, whereas Inductor lowers the broadcast expression through a generic pointwise schedule and current generated code can elide that intermediate cast before the residual add; Inductor cannot do this today because its pointwise codegen does not have a guarded per-channel batchnorm-affine residual lowering that preserves the captured cast boundary while specializing the channel broadcast indexing; the fix is NEW_PATTERN: add a layout-specialized BN-inference residual pointwise lowering for contiguous NCHW tensors with the exact captured cast and sqrt boundaries."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bn_residual_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c = (offsets // HW) % C

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered = x - mean
    denom = libdevice.sqrt(var + 1.0e-5)
    invstd = tl.full((BLOCK,), 1, tl.int32) / denom
    affine = (centered * invstd) * 1.0
    affine = (affine * weight) + bias
    affine = affine.to(tl.bfloat16).to(tl.float32)
    out = residual + affine.to(tl.float32)
    tl.store(out_ptr + offsets, out, mask=mask)


def _launch(inputs, *, BLOCK: int):
    mean, x, var, weight, bias, residual = inputs
    total = x.numel()
    c = int(x.shape[1])
    hw = int(x.shape[2] * x.shape[3])
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    _bn_residual_kernel[(triton.cdiv(total, BLOCK),)](
        mean,
        x,
        var,
        weight,
        bias,
        residual,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return out


@oracle_impl(hardware="B200", point="f49a1957", BLOCK=256)
@oracle_impl(hardware="B200", point="c7ba4cb4", BLOCK=256)
@oracle_impl(hardware="B200", point="896739fb", BLOCK=256)
@oracle_impl(hardware="B200", point="6e9d37a0", BLOCK=256)
@oracle_impl(hardware="B200", point="0d54f7c1", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    return _launch(inputs, BLOCK=BLOCK)
