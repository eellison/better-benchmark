"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileBERT bf16 ReLU LayerNorm scope in one Triton row kernel, including the flat-to-`[256,128,512]` view, bf16 ReLU boundary before fp32 promotion, population `var_mean(..., dim=2, correction=0, keepdim=True)`, returned fp32 mean and `rsqrt(var + 1e-12)` tensors, fp32 affine epilogue, final bf16 `[32768,512]` view output, and returned `[512,32768]` permute alias from the same storage, whereas Inductor lowers the ReLU producer, normalization reduction, affine/cast store, returned statistics, and alias return through generic normalization-template fragments; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization scheduler does not keep the pre-reduction ReLU, saved mean/rsqrt tensors, bf16 affine view, and transpose alias resident in one row schedule while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to fuse ReLU, row statistics, saved-stat side outputs, affine bf16 cast/view, and transpose-alias returns into one full-scope row plan."""

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
    mean_ptr,
    rsqrt_ptr,
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

    tl.store(mean_ptr + rows, mean, mask=rows < ROWS)
    tl.store(rsqrt_ptr + rows, invstd, mask=rows < ROWS)
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


# 354c2fbf: (T([32768,512], bf16), T([512], f32), T([512], f32), S([256,128,512]), S([32768,512]))
@oracle_impl(hardware="B200", point="354c2fbf", BLOCK_H=512, ROW_BLOCK=2, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    x, weight, bias, shape0, shape1 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape1)
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    stat_shape = (norm_shape[0], norm_shape[1], 1)

    mean = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=x.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _relu_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        x,
        weight,
        bias,
        mean,
        rsqrt,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return mean, rsqrt, out, out.permute(1, 0)
