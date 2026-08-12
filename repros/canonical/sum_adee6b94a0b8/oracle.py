"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeepRecommender bf16 SELU-style producer once, stores the hidden contiguous bf16 base needed by the returned transpose view, and accumulates the sibling column sum from the bf16-rounded producer values in one Triton kernel, whereas Inductor schedules the materialized pointwise producer and reduction consumer as generic work over the same expression. Inductor cannot do this today because its scheduler does not fuse a required layout-changing alias output with a compatible sibling reduction while preserving the explicit bf16 producer boundary and final bf16 sum cast. The fix is SCHEDULER_FUSION: teach scheduler/codegen to emit a multi-output pointwise-plus-column-reduction plan for shared producers with side-output views and dtype-cast reduction boundaries."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 256
COLS = 512


@triton.jit
def _selu_bf16_sum_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    sum_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_start = tl.program_id(0) * BLOCK_N
    rows = tl.arange(0, ROWS_)[:, None]
    cols = col_start + tl.arange(0, BLOCK_N)[None, :]
    valid = cols < COLS_
    offsets = rows * COLS_ + cols

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    negative = (x * 1.0) * 1.7580993408473766
    negative = negative * libdevice.exp(gate * 1.0)
    positive = x * 1.0507009873554805
    value = tl.where(gate <= 0.0, negative, positive)
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + offsets, value_bf16, mask=valid)
    col_sum = tl.sum(tl.where(valid, value_bf16.to(tl.float32), 0.0), axis=0)
    rounded_sum = col_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    sum_cols = col_start + tl.arange(0, BLOCK_N)
    tl.store(sum_ptr + sum_cols, rounded_sum, mask=sum_cols < COLS_)


# torchbench_nvidia_deeprecommender train, bf16 [256,512] SELU-style backward fragment.
@oracle_impl(hardware="B200", point="a7f39cdb", BLOCK_N=2, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
    x, gate, shape_param = inputs

    base = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    _selu_bf16_sum_kernel[(triton.cdiv(COLS, BLOCK_N),)](
        x,
        gate,
        base,
        sum_out,
        ROWS_=ROWS,
        COLS_=COLS,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return base.permute(1, 0), sum_out
