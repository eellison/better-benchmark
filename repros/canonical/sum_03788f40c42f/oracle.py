"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 tanh-GELU backward producer once, stores the returned bf16 `[M, 3072]` output, accumulates fp32 row-block column partials from those bf16-rounded producer values, and applies the captured final bf16-to-f32 rounding on the sum output, whereas Inductor lowers the returned producer and sibling column sum through generic pointwise/reduction scheduling with separate materialization and reduction work; Inductor cannot do this today because its scheduler does not form one multi-output producer-store plus partial-reduction plan that preserves the intermediate bf16 rounding boundaries for a value that is both returned and reduced; the fix is SCHEDULER_FUSION: teach multi-output reduction scheduling to generate a fused store-and-partial-reduce template for returned bf16 producers with exact dtype-boundary handling."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


COLS = 3072


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

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        half_x_bf16 = (x * 0.5).to(tl.bfloat16, fp_downcast_rounding="rtne")
        mul_1 = grad * half_x_bf16.to(tl.float32)
        x2 = x * x
        x3 = x2 * x
        tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
        tanh_val = libdevice.tanh(tanh_arg)
        mul_4 = grad * (tanh_val + 1.0)
        sub = 1.0 - tanh_val * tanh_val
        mul_7 = (mul_1 * sub) * 0.7978845608028654
        convert_3 = mul_7.to(tl.bfloat16, fp_downcast_rounding="rtne")
        mul_10 = (mul_7 * 0.044715) * (x2 * 3.0)
        convert_4 = mul_10.to(tl.bfloat16, fp_downcast_rounding="rtne")
        add_2 = (convert_3.to(tl.float32) + convert_4.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        convert_2 = mul_4.to(tl.bfloat16, fp_downcast_rounding="rtne")
        mul_11 = (convert_2.to(tl.float32) * 0.5).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        out_bf16 = (add_2.to(tl.float32) + mul_11.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )

        tl.store(out_ptr + offsets, out_bf16, mask=mask)
        acc += tl.where(mask, out_bf16.to(tl.float32), 0.0)

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
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    out_cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(sum_out_ptr + out_cols, rounded, mask=out_cols < cols)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 05a5e903: (T([16384,3072], bf16), T([16384,3072], bf16), S([32,512,3072]), ...)
@oracle_impl(hardware="B200", point="05a5e903", ROWS_PER_GROUP=256, X_BLOCK=32, R_BLOCK=16, num_warps=8)
# 58ff2bc5: (T([8192,3072], bf16), T([8192,3072], bf16), S([8,1024,3072]), ...)
@oracle_impl(hardware="B200", point="58ff2bc5", ROWS_PER_GROUP=256, X_BLOCK=32, R_BLOCK=16, num_warps=8)
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
        dtype=torch.bfloat16,
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
    return out, sum_out
