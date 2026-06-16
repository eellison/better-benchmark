"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeiT bf16 divide/transpose/sum scope in one Triton column-reduction kernel, including the visible bf16 `[128,1000]` divide output, the returned `[1000,128]` permute alias, the fp32 dim-0 sum over bf16-rounded divide values, and the final bf16-to-fp32 rounding boundary duplicated into two separate `[1000]` outputs, whereas Inductor lowers the captured divide, view, reduction, and cast chain through generic scheduling; Inductor cannot do this today because the scheduler does not fuse a user-visible materialized producer with its compatible reduction consumer while preserving low-precision rounding boundaries and sibling outputs; the fix is SCHEDULER_FUSION: allow a full-scope reduction template to emit the layout-visible producer and all dependent casted reductions from the same loaded tile."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128
COLS = 1000


@triton.jit
def _divide_sum_kernel(
    x_ptr,
    div_ptr,
    sum0_ptr,
    sum1_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, ROWS)
    mask = cols < COLS
    offsets = rows[:, None] * COLS + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask[None, :], other=0.0).to(tl.float32)
    div_bf16 = (x * 0.5).to(tl.bfloat16)
    tl.store(div_ptr + offsets, div_bf16, mask=mask[None, :])

    sums = tl.sum(div_bf16.to(tl.float32), axis=0)
    rounded = sums.to(tl.bfloat16).to(tl.float32)
    tl.store(sum0_ptr + cols, rounded, mask=mask)
    tl.store(sum1_ptr + cols, rounded, mask=mask)


@oracle_impl(hardware="B200", point="e781aba9", BLOCK_N=16, num_warps=2)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, _shape_param_0 = inputs
    del _shape_param_0

    div = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum0 = torch.empty_strided((COLS,), (1,), device=arg0_1.device, dtype=torch.float32)
    sum1 = torch.empty_strided((COLS,), (1,), device=arg0_1.device, dtype=torch.float32)

    _divide_sum_kernel[(triton.cdiv(COLS, BLOCK_N),)](
        arg0_1,
        div,
        sum0,
        sum1,
        ROWS=ROWS,
        COLS=COLS,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return div, div.permute(1, 0), sum0, sum1
