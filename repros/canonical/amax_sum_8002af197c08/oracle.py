"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ConvBERT inference width-9 bias-add softmax scope in one Triton row kernel, matching the captured Inductor fused formulation that views the bf16 `[16384,54]` input as `[98304,9,1]`, adds the bf16 channel bias in the fp32 row-softmax producer, performs stable fp32 amax/libdevice.exp/sum/div, and stores the returned contiguous bf16 probability tensor; Inductor already fuses the graph but lowers it through a generic small-reduction schedule rather than a reusable width-9 softmax template, so the fix is NEW_PATTERN: add a shape-guarded row-softmax lowering with the channel-modulo bias indexing and final bf16 store in one tile."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bias_width9_softmax_out_kernel(
    x_ptr,
    bias_ptr,
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
    scores = x + bias
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(out_ptr + offsets, probs.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@oracle_impl(hardware="B200", point="5fad102b", BLOCK_M=128, BLOCK_N=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    x, bias, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    n_rows = int(out_shape[0])
    out = torch.empty_strided(out_shape, (9, 1, 1), device=x.device, dtype=torch.bfloat16)

    _bias_width9_softmax_out_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        x,
        bias,
        out,
        N_ROWS=n_rows,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
