"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 ALBERT attention softmax scope in one Triton row kernel, including the shape-param view, fp32 stable last-dimension amax/libdevice.exp/sum/div, explicit bf16 rounding, `eq(-inf)`/`any` all-masked-row fallback to the returned zero tensor, the returned 4D zero tensor, the returned 4D softmax tensor, and the returned 3D view alias, whereas Inductor lowers the decomposed view/cast/amax/sub/exp/sum/div/cast/eq/logical_not/any/full/where/expand/view graph as generic reduction and pointwise fragments; Inductor cannot do this today because its pattern library does not recognize this all--inf-safe bf16 softmax with a sibling zero output and layout-only alias output as one full-scope row template; the fix is NEW_PATTERN: add an Inductor lowering that fuses row-validity detection, stable softmax reductions, bf16 cast, zero fallback/full emission, and metadata-only view emission into one generated kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _zero_bf16_kernel(
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    zeros = tl.zeros((BLOCK,), dtype=tl.float32).to(tl.bfloat16)
    tl.store(out_ptr + offsets, zeros, mask=mask)


@triton.jit
def _bf16_softmax_zero_scope_kernel(
    x_ptr,
    where_ptr,
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
    probs = (numer / denom[:, None]).to(tl.bfloat16)
    zeros = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32).to(tl.bfloat16)
    out = tl.where(has_any[:, None], probs, zeros)

    tl.store(where_ptr + offsets, out, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= int(size)
    return tuple(reversed(stride))


# 9135f859: (T([512, 512, 512], bf16), S([8,64,512,512]), S([8,64,512,512]), S([8,64,512,512]), S([512,512,512]))
@oracle_impl(
    hardware="B200",
    point="9135f859",
    BLOCK_M=4,
    BLOCK_N=512,
    ZERO_BLOCK=1024,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, ZERO_BLOCK: int, num_warps: int):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_2

    full_shape = tuple(int(dim) for dim in _shape_param_1)
    view_shape = tuple(int(dim) for dim in _shape_param_3)
    full_stride = _contiguous_stride(full_shape)
    view_stride = _contiguous_stride(view_shape)
    k_len = int(full_shape[-1])
    n_rows = arg0_1.numel() // k_len

    full = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    where = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(n_rows, BLOCK_M),)
    _zero_bf16_kernel[(triton.cdiv(full.numel(), ZERO_BLOCK),)](
        full,
        full.numel(),
        BLOCK=ZERO_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _bf16_softmax_zero_scope_kernel[grid](
        arg0_1,
        where,
        n_rows=n_rows,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )

    return full, where, torch.as_strided(where, view_shape, view_stride)
