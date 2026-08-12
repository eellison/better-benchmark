"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GoogleFnet fp32 tanh-approximate GELU plus hidden-size-768 LayerNorm inference scope in one Triton row kernel, including the metadata-only `[16384,768] -> [32,512,768]` view, Inductor-order `0.5*x*(1+tanh(0.7978845608028654*(x+0.044715*x*x*x)))`, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, affine scale/bias, and final contiguous `[16384,768]` output view, whereas Inductor lowers the GELU producer, normalization reduction, and affine epilogue through generic pointwise and normalization-template scheduling; Inductor cannot fuse this full returned-output envelope today because the fixed-hidden normalization scheduler does not keep the tanh-GELU producer values resident across row statistics and affine output stores while preserving fp32 arithmetic boundaries; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline tanh-GELU producers into the row reduction and affine epilogue for fixed transformer hidden sizes."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-12


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
def _gelu_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    valid = (row_offsets[:, None] < ROWS) & (cols < HIDDEN)
    offsets = rows * HIDDEN + cols

    x = tl.load(
        x_ptr + offsets,
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    half = _f32_mul(x, 0.5)
    x2 = _f32_mul(x, x)
    x3 = _f32_mul(x2, x)
    cubic = _f32_mul(x3, 0.044715)
    tanh_arg = _f32_mul(_f32_add(x, cubic), 0.7978845608028654)
    gelu = _f32_mul(half, _f32_add(libdevice.tanh(tanh_arg), 1.0))

    reduce_input = tl.where(valid, gelu, 0.0)
    mean = tl.sum(reduce_input, axis=1) / HIDDEN
    centered = _f32_sub(gelu, mean[:, None])
    centered_masked = tl.where(valid, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    normalized = _f32_mul(centered, invstd[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    out = _f32_add(_f32_mul(normalized, weight), bias)
    tl.store(out_ptr + offsets, out, mask=valid)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="95772c51", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, _shape0, shape1 = inputs
    out_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg1_1.shape[0])

    output = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    _gelu_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        output,
        ROWS=rows,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
