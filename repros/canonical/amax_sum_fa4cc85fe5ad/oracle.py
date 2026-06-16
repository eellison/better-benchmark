"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete T5/MT5 bf16 attention softmax scope in one Triton row kernel, including the shape-param view, bf16-to-fp32 promotion, stable last-dimension amax/libdevice.exp/sum/div, explicit bf16 probability cast, expand, and final contiguous view, whereas Inductor lowers the decomposed view/cast/amax/sub/exp/sum/div/cast/expand/view graph through its generic reduction scheduling; Inductor cannot do this today for the large T5 point because the scheduler/codegen does not select the same fixed-width row-softmax template after the view-only and expand-only graph fragments while preserving the bf16 input and output cast boundaries; the fix is SCHEDULER_FUSION: add or retune a guarded bf16 row-softmax lowering that sinks the layout-only operations into the store and uses the shape-specialized K=128/K=1024 schedules."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bf16_softmax_kernel(
    x_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < N_COLS
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * N_COLS + cols[None, :]

    scores = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(scores, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(out_ptr + offsets, probs.to(tl.bfloat16), mask=mask)


def _launch_softmax(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_2),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_cols = int(x.shape[-1])
    n_rows = x.numel() // n_cols

    _bf16_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        x,
        out,
        N_ROWS=n_rows,
        N_COLS=n_cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


# 215d1cc0: (T([192,128,128], bf16), S([32,6,128,128]), S([32,6,128,128]), S([192,128,128]))
@oracle_impl(hardware="B200", point="215d1cc0", BLOCK_M=16, BLOCK_N=128, num_warps=4)
# 5b248c57: (T([64,1024,1024], bf16), S([8,8,1024,1024]), S([8,8,1024,1024]), S([64,1024,1024]))
@oracle_impl(hardware="B200", point="5b248c57", BLOCK_M=1, BLOCK_N=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    return _launch_softmax(inputs, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps)
