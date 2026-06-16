"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ConvBert width-9 bias-add softmax scope in one Triton row kernel, including the explicit fp32-to-bf16 bias cast, fp32 score construction from the bf16 matrix and rounded bias value, stable fp32 amax/libdevice.exp/sum/div, returned f32 max and denominator side outputs, and final contiguous bf16 probability tensor; Inductor already fuses the graph but only through a generic persistent reduction, not a reusable small-K row-softmax template; the fix is NEW_PATTERN: add a guarded width-9 softmax lowering that emits the f32 side reductions and bf16 probabilities from one tile with the shape-specialized indexing."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bias_width9_softmax_kernel(
    bias_ptr,
    x_ptr,
    max_ptr,
    sum_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < 9
    mask = row_mask[:, None] & col_mask[None, :]

    offsets = rows[:, None] * 9 + cols[None, :]
    channels = offsets % 54

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    scores = x + _round_to_bf16_f32(bias)
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(max_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)
    tl.store(out_ptr + offsets, probs.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@oracle_impl(hardware="B200", point="6210275d", BLOCK_M=128, BLOCK_N=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    bias, x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    n_rows = int(_shape_param_2[0])
    max_out = torch.empty_strided((n_rows, 1, 1), (1, 1, 1), device=x.device, dtype=torch.float32)
    sum_out = torch.empty_strided((n_rows, 1, 1), (1, 1, 1), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(int(dim) for dim in _shape_param_2), (9, 1, 1), device=x.device, dtype=torch.bfloat16)

    _bias_width9_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        bias,
        x,
        max_out,
        sum_out,
        out,
        N_ROWS=n_rows,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return max_out, sum_out, out
