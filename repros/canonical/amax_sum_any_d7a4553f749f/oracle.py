"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 ALBERT attention softmax scope in one Triton row kernel, including the shape-param view, fp32 stable last-dimension amax/libdevice.exp/sum/div, explicit bf16 rounding, `eq(-inf)`/`any` all-masked-row fallback to arg1, the returned 4D tensor, and the returned 3D view alias, whereas Inductor lowers the decomposed view/cast/amax/sub/exp/sum/div/cast/eq/logical_not/any/where/expand/view graph as generic reduction and pointwise fragments; Inductor cannot do this today because its pattern library does not recognize this all--inf-safe bf16 softmax with a tensor fallback and sibling layout-only output as one full-scope row template; the fix is NEW_PATTERN: add an Inductor lowering that fuses row-validity detection, stable softmax reductions, bf16 cast, fallback selection, and metadata-only view emission into one generated kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bf16_softmax_fallback_kernel(
    x_ptr,
    fallback_ptr,
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

    fallback = tl.load(fallback_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.where(has_any[:, None], probs, fallback)
    tl.store(out_ptr + offsets, out, mask=mask)


# b31e9601: (T([512, 512, 512], bf16), T([8, 64, 512, 512], bf16), ...)
@oracle_impl(hardware="B200", point="b31e9601", BLOCK_M=4, BLOCK_N=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    out_3d_shape = tuple(int(dim) for dim in _shape_param_2)
    k_len = int(arg0_1.shape[-1])
    n_rows = arg0_1.numel() // k_len

    out_3d = torch.empty_strided(
        out_3d_shape,
        (out_3d_shape[1] * out_3d_shape[2], out_3d_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(n_rows, BLOCK_M),)
    _bf16_softmax_fallback_kernel[grid](
        arg0_1,
        arg1_1,
        out_3d,
        n_rows=n_rows,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )

    return out_3d.view_as(arg1_1), out_3d
