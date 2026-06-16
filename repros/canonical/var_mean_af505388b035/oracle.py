"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 exact-erf GELU plus population LayerNorm scope in one Triton row kernel, including the metadata-only input view, fp32 GELU, required bf16 GELU rounding before the statistics, correction=0 var_mean over the hidden dimension, eps=1e-12 rsqrt, bf16 affine scale/bias, final bf16 cast, and flattened output view, whereas Inductor lowers the decomposed GELU/cast/var_mean/rsqrt/affine/view graph through its generic fused normalization schedule; Inductor cannot do this today because the normalization scheduler does not have a guarded exact-GELU-producer LayerNorm specialization that keeps the bf16-rounded activation tile resident across the row moments and affine epilogue for fixed hidden sizes; the fix is SCHEDULER_FUSION: add a bf16 exact-GELU-producing LayerNorm template that fuses the pointwise producer, cast boundary, row moments, epsilon rsqrt, affine epilogue, and final viewed store."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gelu_layernorm_bf16_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < HIDDEN)
    offsets = rows[:, None] * HIDDEN + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    erf_term = libdevice.erf(x * 0.7071067811865476) + 1.0
    gelu = ((x * 0.5) * erf_term).to(tl.bfloat16).to(tl.float32)

    col_mask = cols[None, :] < HIDDEN
    mean = tl.sum(tl.where(col_mask, gelu, 0.0), axis=1) / HIDDEN
    centered = gelu - mean[:, None]
    variance = tl.sum(tl.where(col_mask, centered * centered, 0.0), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(variance + 1.0e-12)

    weight = tl.load(weight_ptr + cols, mask=cols < HIDDEN, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=cols < HIDDEN, other=0.0).to(tl.float32)
    y = centered * invstd[:, None] * weight[None, :] + bias[None, :]
    tl.store(out_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# 5428f51a: (T([16384,768], bf16), T([768], bf16), T([768], bf16), ...)
@oracle_impl(hardware="B200", point="5428f51a", BLOCK_M=1, BLOCK_H=1024, num_warps=1)
# 113efab4: (T([4096,1536], bf16), T([1536], bf16), T([1536], bf16), ...)
@oracle_impl(hardware="B200", point="113efab4", BLOCK_M=1, BLOCK_H=2048, num_warps=8)
# f4120204: (T([32768,768], bf16), T([768], bf16), T([768], bf16), ...)
@oracle_impl(hardware="B200", point="f4120204", BLOCK_M=1, BLOCK_H=1024, num_warps=4)
# bc0fb1fb: (T([32768,128], bf16), T([128], bf16), T([128], bf16), ...)
@oracle_impl(hardware="B200", point="bc0fb1fb", BLOCK_M=4, BLOCK_H=128, num_warps=4)
# 6462934d: (T([8192,768], bf16), T([768], bf16), T([768], bf16), ...)
@oracle_impl(hardware="B200", point="6462934d", BLOCK_M=1, BLOCK_H=1024, num_warps=4)
# 7648de68: (T([8192,1024], bf16), T([1024], bf16), T([1024], bf16), ...)
@oracle_impl(hardware="B200", point="7648de68", BLOCK_M=1, BLOCK_H=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out = torch.empty_like(arg0_1)
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)
    _gelu_layernorm_bf16_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return out.view(tuple(int(dim) for dim in _shape_param_1))
