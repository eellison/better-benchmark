"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ALBERT bf16 tanh-GELU producer plus hidden-size-128 LayerNorm training scope in one Triton row kernel, including the metadata-only `[4096,128] -> [8,512,128]` view, the captured bf16 `view * 0.5` boundary, fp32 `x + 0.044715*x^3`, natural `tanh(0.7978845608028654 * ...)`, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, returned mean and rsqrt side outputs, affine scale/bias epilogue, final bf16 cast, and contiguous `[4096,128]` view, whereas Inductor lowers the GELU producer, row statistics, affine cast, and statistic side outputs through generic pointwise and normalization-template scheduling; Inductor cannot fuse this full returned-output envelope today because the fixed-hidden normalization scheduler does not keep the dtype-boundary-sensitive tanh-GELU producer resident across row statistics, side-output stores, and affine output stores; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline tanh-GELU producers with captured bf16 roundings and emit mean, rsqrt, and final bf16 view from one row plan."""

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
    mean_ptr,
    rsqrt_ptr,
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

    x_bf16 = tl.load(
        x_ptr + offsets,
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    )
    x = x_bf16.to(tl.float32)

    half = _f32_mul(x, 0.5).to(tl.bfloat16).to(tl.float32)
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

    tl.store(mean_ptr + row_offsets, mean, mask=row_offsets < ROWS)
    tl.store(rsqrt_ptr + row_offsets, invstd, mask=row_offsets < ROWS)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=valid)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="a0e6bc91", BLOCK_H=128, ROW_BLOCK=4, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    stat_shape = (view_shape[0], view_shape[1], 1)
    stat_stride = (view_shape[1], 1, 1)

    mean = torch.empty_strided(
        stat_shape,
        stat_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        stat_shape,
        stat_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _gelu_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        mean,
        rsqrt,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return mean, rsqrt, out
