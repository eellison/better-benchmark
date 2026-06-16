"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 `[8192, 262144]` row-wise stable softmax, including fp32 input conversion, row max, exact `exp`, row sum, normalization, and bf16 output materialization in one Triton lowering, whereas Inductor lowers the amax/exp/sum/div chain through its generic large-row softmax reduction schedule; Inductor cannot do this today because scheduler/codegen has no dedicated very-wide static softmax lowering that keeps the row statistics scalar while streaming the output without materializing fp32 intermediates; the fix is NEW_PATTERN: add a static large-vocabulary softmax lowering that uses a shape-specialized online reduction plus normalization pass."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 8192
COLS = 262144


@triton.jit
def _online_softmax_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = row * N
    cols = tl.arange(0, BLOCK_N)

    running_max = -float("inf")
    running_sum = 0.0
    for start in tl.range(0, N, BLOCK_N):
        offsets = start + cols
        x = tl.load(x_ptr + row_base + offsets).to(tl.float32)
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        running_max = new_max

    inv_sum = 1.0 / running_sum
    for start in tl.range(0, N, BLOCK_N):
        offsets = start + cols
        x = tl.load(x_ptr + row_base + offsets).to(tl.float32)
        out = tl.exp(x - running_max) * inv_sum
        tl.store(out_ptr + row_base + offsets, out.to(tl.bfloat16))


@oracle_impl(
    hardware="B200",
    point="4ac62a08",
    BLOCK_N=8192,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    x = inputs[0]
    out = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _online_softmax_kernel[(ROWS,)](
        x,
        out,
        N=COLS,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
