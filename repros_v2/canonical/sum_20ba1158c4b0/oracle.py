"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete TTS Angular guarded row reduction, bf16 divide/add epilogue, zero-filled `[64, 50, 256]` materialization, and `select_scatter(..., dim=1, index=-1)` store in one Triton kernel, whereas Inductor lowers the row reduction, bf16 casts, final bf16 add, and full/select_scatter materialization as separate generic schedules; Inductor cannot do this today because scheduler/codegen does not model a row-reduction producer feeding a structured zero-fill select_scatter side output as one fused scatter-reduce template while preserving the explicit bf16 rounding boundaries; the fix is SCATTER_REDUCE: add a structured select_scatter-reduce lowering that keeps the row sum in registers while writing the sparse slice and required zero materialization directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 64
DEPTH = 50
COLS = 256
SCATTER_INDEX = DEPTH - 1


@triton.jit
def _select_scatter_reduce_bf16_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    out_ptr,
    ARG0_ROW_STRIDE: tl.constexpr,
    ARG0_COL_STRIDE: tl.constexpr,
    ARG1_ROW_STRIDE: tl.constexpr,
    ARG2_ROW_STRIDE: tl.constexpr,
    ARG2_COL_STRIDE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    OUT_DEPTH_STRIDE: tl.constexpr,
    OUT_COL_STRIDE: tl.constexpr,
    DEPTH_: tl.constexpr,
    COLS_: tl.constexpr,
    SCATTER_INDEX_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < COLS_

    arg0 = tl.load(
        arg0_ptr + row * ARG0_ROW_STRIDE + cols * ARG0_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    arg1 = tl.load(arg1_ptr + row * ARG1_ROW_STRIDE).to(tl.float32)
    arg2 = tl.load(
        arg2_ptr + row * ARG2_ROW_STRIDE + cols * ARG2_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    denom = tl.where(arg1 < 1.0e-12, 1.0e-12, arg1)
    row_terms = (-arg0) * ((arg2 / denom) / denom)
    row_sum = tl.sum(row_terms, axis=0)

    div2_bf16 = (arg0 / denom).to(tl.bfloat16, fp_downcast_rounding="rtne")
    guarded_sum = tl.where(arg1 >= 1.0e-12, row_sum, 0.0)
    raw_div = arg2 / arg1
    guarded_div = tl.where(arg1 == 0.0, 0.0, raw_div)
    mul1_bf16 = (guarded_sum * guarded_div).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    scattered = (div2_bf16.to(tl.float32) + mul1_bf16.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    zero = tl.zeros((BLOCK_N,), dtype=tl.float32).to(tl.bfloat16)

    for depth in tl.static_range(0, DEPTH_):
        values = scattered if depth == SCATTER_INDEX_ else zero
        tl.store(
            out_ptr
            + row * OUT_ROW_STRIDE
            + depth * OUT_DEPTH_STRIDE
            + cols * OUT_COL_STRIDE,
            values,
            mask=mask,
        )


# ee4b9eab: torchbench TTS Angular train, f32 row reduction into bf16 select_scatter.
@oracle_impl(hardware="B200", point="ee4b9eab", BLOCK_N=256, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    out = torch.empty_strided(
        (ROWS, DEPTH, COLS),
        (DEPTH * COLS, COLS, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _select_scatter_reduce_bf16_kernel[(ROWS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        ARG0_ROW_STRIDE=arg0_1.stride(0),
        ARG0_COL_STRIDE=arg0_1.stride(1),
        ARG1_ROW_STRIDE=arg1_1.stride(0),
        ARG2_ROW_STRIDE=arg2_1.stride(0),
        ARG2_COL_STRIDE=arg2_1.stride(1),
        OUT_ROW_STRIDE=out.stride(0),
        OUT_DEPTH_STRIDE=out.stride(1),
        OUT_COL_STRIDE=out.stride(2),
        DEPTH_=DEPTH,
        COLS_=COLS,
        SCATTER_INDEX_=SCATTER_INDEX,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
