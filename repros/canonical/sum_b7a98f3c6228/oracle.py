"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeepRecommender bf16 SELU-style producer once, writes both required zero-padded layouts, and accumulates the sibling bf16-rounded column sum from the same producer tile, whereas Inductor schedules the pointwise activation, right-pad materialization, transpose-pad materialization, and column reduction as separate generic work over the same large `[256,197951]` expression; Inductor cannot do this today because its scheduler does not fuse multiple layout-changing padded side outputs with a compatible sibling reduction while preserving the explicit bf16 rounding boundary; the fix is SCHEDULER_FUSION: teach scheduler/codegen to emit a multi-output pointwise-plus-column-reduction plan for shared producers with static pad epilogues and dtype-cast reduction boundaries."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 256
COLS = 197951
PADDED_COLS = 197952
GATE_STRIDE_ROW = 197952


@triton.jit
def _selu_materialize_kernel(
    x_ptr,
    gate_ptr,
    out_right_pad_ptr,
    ROWS_N: tl.constexpr,
    COLS_N: tl.constexpr,
    PADDED_COLS_N: tl.constexpr,
    GATE_STRIDE_ROW_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    active = cols < COLS_N

    x_offsets = row * COLS_N + cols
    gate_offsets = row * GATE_STRIDE_ROW_N + cols
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=active, other=0.0).to(tl.float32)

    neg = (x * 1.0) * 1.7580993408473766
    neg = neg * libdevice.exp(gate * 1.0)
    pos = x * 1.0507009873554805
    value = tl.where(gate <= 0.0, neg, pos)
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_right_pad_ptr + row * PADDED_COLS_N + cols, value_bf16, mask=active)


@triton.jit
def _transpose_sum_kernel(
    out_right_pad_ptr,
    out_transpose_pad_ptr,
    sum_ptr,
    ROWS_N: tl.constexpr,
    COLS_N: tl.constexpr,
    PADDED_COLS_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_start = tl.program_id(0) * BLOCK_N
    rows = tl.arange(0, ROWS_N)[:, None]
    cols = col_start + tl.arange(0, BLOCK_N)[None, :]
    active = cols < COLS_N

    value_bf16 = tl.load(
        out_right_pad_ptr + rows * PADDED_COLS_N + cols,
        mask=active,
        other=0.0,
    )
    tl.store(out_transpose_pad_ptr + cols * ROWS_N + rows, value_bf16, mask=active)
    col_sum = tl.sum(tl.where(active, value_bf16.to(tl.float32), 0.0), axis=0)
    rounded_sum = col_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    sum_cols = col_start + tl.arange(0, BLOCK_N)
    tl.store(sum_ptr + sum_cols, rounded_sum, mask=sum_cols < COLS_N)


@triton.jit
def _zero_pad_tail_kernel(
    out_right_pad_ptr,
    out_transpose_pad_ptr,
    ROWS_N: tl.constexpr,
    COLS_N: tl.constexpr,
    PADDED_COLS_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_R)
    active = offsets < ROWS_N
    zero = tl.full((BLOCK_R,), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_right_pad_ptr + offsets * PADDED_COLS_N + COLS_N, zero, mask=active)
    tl.store(out_transpose_pad_ptr + COLS_N * ROWS_N + offsets, zero, mask=active)


# 348dd978: nvidia_deeprecommender train, bf16 [256,197951] with padded second input stride.
@oracle_impl(
    hardware="B200",
    point="348dd978",
    MAT_BLOCK_N=2048,
    REDUCE_BLOCK_N=32,
    mat_warps=4,
    reduce_warps=8,
)
def oracle_forward(
    inputs,
    *,
    MAT_BLOCK_N: int,
    REDUCE_BLOCK_N: int,
    mat_warps: int,
    reduce_warps: int,
):
    x, gate, _shape = inputs
    del _shape

    out_right_pad = torch.empty_strided(
        (ROWS, PADDED_COLS),
        (PADDED_COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out_transpose_pad = torch.empty_strided(
        (PADDED_COLS, ROWS),
        (ROWS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        (COLS,),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    _selu_materialize_kernel[(ROWS, triton.cdiv(COLS, MAT_BLOCK_N))](
        x,
        gate,
        out_right_pad,
        ROWS_N=ROWS,
        COLS_N=COLS,
        PADDED_COLS_N=PADDED_COLS,
        GATE_STRIDE_ROW_N=GATE_STRIDE_ROW,
        BLOCK_N=MAT_BLOCK_N,
        num_warps=mat_warps,
        num_stages=3,
    )
    _transpose_sum_kernel[(triton.cdiv(COLS, REDUCE_BLOCK_N),)](
        out_right_pad,
        out_transpose_pad,
        sum_out,
        ROWS_N=ROWS,
        COLS_N=COLS,
        PADDED_COLS_N=PADDED_COLS,
        BLOCK_N=REDUCE_BLOCK_N,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _zero_pad_tail_kernel[(1,)](
        out_right_pad,
        out_transpose_pad,
        ROWS_N=ROWS,
        COLS_N=COLS,
        PADDED_COLS_N=PADDED_COLS,
        BLOCK_R=ROWS,
        num_warps=8,
        num_stages=3,
    )
    return out_right_pad, out_transpose_pad, sum_out
