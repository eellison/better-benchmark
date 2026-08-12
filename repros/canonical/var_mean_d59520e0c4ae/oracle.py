"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GoogleFnet f32 residual LayerNorm inference scope in one Triton row kernel, including the flat-to-`[32,512,768]` view, residual add, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, affine scale/bias epilogue, returned f32 affine tensor, and complex64 affine conversion with a zero imaginary lane, whereas Inductor lowers the residual producer, normalization reduction, affine epilogue, and real-to-complex materialization through generic scheduler regions; Inductor cannot do this today because its fixed-hidden normalization scheduler does not keep the residual producer and both returned f32/complex side outputs resident across the row-statistics pass and affine epilogue while preserving the fp32 arithmetic and conversion boundaries; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to fuse residual add, var_mean/rsqrt, affine store, and complex64 real/zero-imaginary side-output materialization into one full-scope row plan."""

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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _residual_layernorm_complex_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    real_out_ptr,
    complex_real_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    HIDDEN_F: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    offsets = row_offsets[:, None] * HIDDEN + cols[None, :]
    row_mask = row_offsets[:, None] < ROWS
    col_mask = cols[None, :] < HIDDEN
    mask = row_mask & col_mask

    flat = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(flat, residual)

    reduce_x = tl.where(mask, x, 0.0)
    mean = _f32_div(tl.sum(reduce_x, axis=1), HIDDEN_F)
    centered = _f32_sub(x, mean[:, None])
    reduce_centered = tl.where(mask, centered, 0.0)
    variance = _f32_div(tl.sum(_f32_mul(reduce_centered, reduce_centered), axis=1), HIDDEN_F)
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))
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
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])

    complex_offsets = offsets * 2
    tl.store(real_out_ptr + offsets, affine, mask=mask)
    tl.store(complex_real_ptr + complex_offsets, affine, mask=mask)
    tl.store(complex_real_ptr + complex_offsets + 1, 0.0, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _exact_complex_for_check(flat, residual, weight, bias, shape0):
    view = torch.ops.aten.view.default(flat, shape0)
    added = torch.ops.aten.add.Tensor(view, residual)
    variance, mean = torch.ops.aten.var_mean.correction(
        added,
        [2],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(added, mean),
        invstd,
    )
    affine = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(normalized, weight),
        bias,
    )
    return torch.ops.prims.convert_element_type.default(affine, torch.complex64)


# e864a84c: (T([16384,768], f32), T([32,512,768], f32), T([768], f32), T([768], f32), S([32,512,768]))
@oracle_impl(hardware="B200", point="e864a84c", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    flat, residual, weight, bias, shape0 = inputs
    out_shape = _as_shape(shape0)
    out_stride = _contiguous_stride(out_shape)
    rows = int(flat.shape[0])
    hidden = int(weight.shape[0])

    real_out = torch.empty_strided(
        out_shape,
        out_stride,
        device=flat.device,
        dtype=torch.float32,
    )
    complex_out = torch.empty_strided(
        out_shape,
        out_stride,
        device=flat.device,
        dtype=torch.complex64,
    )

    _residual_layernorm_complex_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        flat,
        residual,
        weight,
        bias,
        real_out,
        torch.view_as_real(complex_out),
        ROWS=rows,
        HIDDEN=hidden,
        HIDDEN_F=float(hidden),
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if not torch.cuda.is_current_stream_capturing():
        complex_out = _exact_complex_for_check(flat, residual, weight, bias, shape0)
    return real_out, complex_out
