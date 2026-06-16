"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 attention softmax scope, including the shape-param view, `eq(-inf)`/`any` all-masked-row guard, fp32 stable last-dimension amax/exp/sum/div, explicit bf16 rounding, zero fill for all--inf rows, expand, and final contiguous view in one Triton row kernel, whereas Inductor lowers the decomposed view/eq/logical_not/any/logical_not/full/cast/amax/sub/exp/sum/div/cast/where/expand/view graph as generic reduction and pointwise fragments; Inductor cannot do this today because its pattern library does not recognize this all--inf-safe returned bf16 softmax as one semantic row template with the layout-only epilogue sunk into the store; the fix is NEW_PATTERN: add an Inductor lowering that fuses row-validity detection, stable softmax reductions, all--inf zeroing, bf16 cast, and final view emission into one generated kernel."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bf16_safe_softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * k_len + cols[None, :]

    scores = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    live = mask & (scores != -float("inf"))
    has_any = tl.max(tl.where(live, 1, 0), axis=1) != 0

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(has_any, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(live, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    denom = tl.where(has_any, denom, 1.0)
    probs = numer / denom[:, None]
    probs = tl.where(has_any[:, None], probs, 0.0)

    tl.store(out_ptr + offsets, probs, mask=mask)


# 9135f859: (T([512, 512, 512], bf16), S([8, 64, 512, 512]), ...)
# 59a0e132: (T([384, 512, 512], bf16), S([32, 12, 512, 512]), ...)
# e6f344ac: (T([512, 128, 128], bf16), S([16, 32, 128, 128]), ...)
# 2fe3efd1: (T([3072, 128, 128], bf16), S([256, 12, 128, 128]), ...)
# 730342a3: (T([256, 512, 512], bf16), S([64, 4, 512, 512]), ...)
# bcf6fe02: (T([1024, 128, 128], bf16), S([64, 16, 128, 128]), ...)
@oracle_impl(hardware="B200", point="9135f859", BLOCK_M=4, BLOCK_N=512, num_warps=8)
@oracle_impl(hardware="B200", point="59a0e132", BLOCK_M=4, BLOCK_N=512, num_warps=4)
@oracle_impl(hardware="B200", point="730342a3", BLOCK_M=4, BLOCK_N=512, num_warps=4)
@oracle_impl(hardware="B200", point="e6f344ac", BLOCK_M=8, BLOCK_N=128, num_warps=4)
@oracle_impl(hardware="B200", point="2fe3efd1", BLOCK_M=8, BLOCK_N=128, num_warps=4)
@oracle_impl(hardware="B200", point="bcf6fe02", BLOCK_M=8, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    x, _shape0, _shape1, _shape2, out_shape = inputs
    k_len = int(x.shape[-1])
    n_rows = x.numel() // k_len
    out_shape = tuple(int(dim) for dim in out_shape)
    out_stride = (out_shape[1] * out_shape[2], out_shape[2], 1)
    out = torch.empty_strided(
        out_shape,
        out_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )
    _bf16_safe_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        x,
        out,
        n_rows=n_rows,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
