"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileBERT bf16 ReLU LayerNorm inference scope in one Triton row kernel, including the flat-to-[256,128,512] view, NaN-preserving bf16 ReLU boundary before fp32 promotion, population var_mean(..., dim=2, correction=0, keepdim=True), eps=1e-12 rsqrt, fp32 affine epilogue with bf16 scale/bias inputs, final bf16 cast, and contiguous [32768,512] view output, whereas Inductor lowers the ReLU producer, row-statistics reduction, affine epilogue, cast, and view store through generic normalization-template fragments; Inductor cannot fuse this exact envelope today because its fixed-hidden normalization scheduler does not keep the bf16 activation producer and final bf16 affine view resident in one row plan while preserving dtype rounding boundaries; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to fuse ReLU, hidden-size-512 var_mean/rsqrt, bf16-parameter affine, and final bf16 view emission into one guarded full-scope row schedule."""

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
def _relu_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_ids = rows[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask
    offsets = row_ids * HIDDEN + cols

    x_bf16 = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    relu_bf16 = tl.where((x_bf16 > 0.0) | (x_bf16 != x_bf16), x_bf16, 0.0).to(
        tl.bfloat16
    )
    x = relu_bf16.to(tl.float32)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))
    normalized = _f32_mul(centered, invstd[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# b4251142: (T([32768,512], bf16), T([512], bf16), T([512], bf16), S([256,128,512]), S([32768,512]))
@oracle_impl(hardware="B200", point="b4251142", BLOCK_H=512, ROW_BLOCK=8, num_warps=4, num_stages=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    x, weight, bias, _shape0, shape1 = inputs
    out_shape = _as_shape(shape1)
    rows = int(x.shape[0])
    hidden = int(x.shape[1])

    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _relu_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        x,
        weight,
        bias,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
