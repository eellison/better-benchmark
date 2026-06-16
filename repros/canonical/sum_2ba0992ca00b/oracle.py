"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete f32 tanh-GELU backward producer once, stores the returned f32 `[16384, 3072]` tensor, returns its metadata-only transpose alias, and accumulates the sibling f32 column sum through row-block partials from the same producer values, whereas Inductor lowers the returned producer, transpose view, and dependent sum through generic pointwise/reduction scheduling with separate materialization and reduction work; Inductor cannot do this today because its scheduler does not form one alias-aware multi-output producer-store plus partial-reduction plan for a value that is both returned and reduced; the fix is SCHEDULER_FUSION: teach multi-output reduction scheduling to generate a fused store-and-partial-reduce template that preserves view aliases from the single materialized producer."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gelu_backward_f32(grad, x):
    grad_f32 = grad.to(tl.float32)
    x_f32 = x.to(tl.float32)

    mul = x_f32 * 0.5
    mul_1 = grad_f32 * mul
    x2 = x_f32 * x_f32
    x3 = x2 * x_f32
    tanh_arg = (x_f32 + x3 * 0.044715) * 0.7978845608028654
    tanh_val = libdevice.tanh(tanh_arg)
    mul_4 = grad_f32 * (tanh_val + 1.0)
    sub = 1.0 - tanh_val * tanh_val
    mul_7 = (mul_1 * sub) * 0.7978845608028654
    mul_10 = (mul_7 * 0.044715) * (x2 * 3.0)
    add_2 = mul_7 + mul_10
    mul_11 = mul_4 * 0.5
    return add_2 + mul_11


@triton.jit
def _store_and_partial_sum_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    partials_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    xnumel: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    X_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[:, None]
    group = xindex // cols
    col = xindex - group * cols
    rbase = tl.arange(0, R_BLOCK)[None, :]
    acc = tl.zeros((X_BLOCK, R_BLOCK), dtype=tl.float32)

    for roffset in tl.range(0, ROWS_PER_GROUP, R_BLOCK):
        rindex = roffset + rbase
        row = group * ROWS_PER_GROUP + rindex
        mask = (xindex < xnumel) & (row < rows)
        offsets = row * cols + col

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = _gelu_backward_f32(grad, x)

        tl.store(out_ptr + offsets, out, mask=mask)
        acc += tl.where(mask, out, 0.0)

    partial = tl.sum(acc, axis=1)[:, None]
    tl.store(partials_ptr + xindex, partial, mask=xindex < xnumel)


@triton.jit
def _final_sum_kernel(
    partials_ptr,
    sum_out_ptr,
    groups: tl.constexpr,
    cols: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_block = tl.program_id(0)
    group_offsets = tl.arange(0, GROUP_BLOCK)[:, None]
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (group_offsets < groups) & (col_offsets < cols)
    vals = tl.load(
        partials_ptr + group_offsets * cols + col_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, vals, 0.0), axis=0)
    out_cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(sum_out_ptr + out_cols, total, mask=out_cols < cols)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 94cc1ac5: (T([16384,3072], f32), T([16384,3072], f32), S([32,512,3072]), ...)
@oracle_impl(hardware="B200", point="94cc1ac5", ROWS_PER_GROUP=256, X_BLOCK=32, R_BLOCK=16, num_warps=8)
def oracle_forward(inputs, *, ROWS_PER_GROUP: int, X_BLOCK: int, R_BLOCK: int, num_warps: int):
    grad, x, _shape0, _shape1, out_shape_arg, sum_shape_arg = inputs
    rows = int(grad.shape[0])
    cols = int(grad.shape[1])
    groups = triton.cdiv(rows, ROWS_PER_GROUP)
    xnumel = groups * cols

    out = torch.empty_strided(
        _shape_tuple(out_shape_arg),
        (cols, 1),
        device=grad.device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (groups, cols),
        (cols, 1),
        device=grad.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        _shape_tuple(sum_shape_arg),
        (1,),
        device=grad.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(xnumel, X_BLOCK),)
    _store_and_partial_sum_kernel[grid](
        grad,
        x,
        out,
        partials,
        rows=rows,
        cols=cols,
        xnumel=xnumel,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        X_BLOCK=X_BLOCK,
        R_BLOCK=R_BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    _final_sum_kernel[(triton.cdiv(cols, 32),)](
        partials,
        sum_out,
        groups=groups,
        cols=cols,
        GROUP_BLOCK=64,
        BLOCK_N=32,
        num_warps=8,
        num_stages=1,
    )
    return out, out.permute(1, 0), sum_out
