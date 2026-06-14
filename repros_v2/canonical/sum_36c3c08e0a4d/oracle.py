"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete LearningToPaint bf16 pointwise-materialization plus sibling column-sum scope in one Triton kernel, including the f32 `arg0 * arg1 * (1 - arg1)` formulation, explicit bf16 producer rounding, the contiguous `[96,65]` output, its returned `[65,96]` transpose alias, and the `[65]` f32 output produced by summing the materialized bf16 values in f32 and bf16-rounding that sum before converting back to f32. Inductor currently schedules the visible materialization and the sibling column reduction as generic work over the same producer, with separate handling for the layout-return and reduction-output consumers; it cannot do this today because the scheduler does not form one full-scope multi-output plan that shares a bf16-rounded pointwise producer across a required backing store, a transpose alias, and a compatible small column reduction. The fix is SCHEDULER_FUSION: teach reduction scheduling to keep small sibling reductions in the same kernel as required materialized side outputs while preserving bf16/f32 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 96
COLS = 65
OUT_SHAPE = (ROWS, COLS)
OUT_STRIDE = (COLS, 1)
PERMUTE_SHAPE = (COLS, ROWS)
PERMUTE_STRIDE = (1, COLS)


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _materialize_and_sum_kernel(
    arg0_ptr,
    arg1_ptr,
    out_ptr,
    sum_ptr,
    ROWS_N: tl.constexpr,
    COLS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    mask = rows < ROWS_N
    offsets = rows * COLS_N + col

    a = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    value = _f32_mul(a, _f32_mul(b, _f32_sub(1.0, b))).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    value = tl.where(mask, value, 0.0)

    tl.store(out_ptr + offsets, value, mask=mask)
    col_sum = tl.sum(value.to(tl.float32), axis=0)
    rounded_sum = col_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + col, rounded_sum)


@oracle_impl(hardware="B200", point="286ee5cf", BLOCK_M=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    arg0, arg1, shape0 = inputs
    del shape0

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    sums = torch.empty_strided((COLS,), (1,), device=arg0.device, dtype=torch.float32)
    _materialize_and_sum_kernel[(COLS,)](
        arg0,
        arg1,
        out,
        sums,
        ROWS_N=ROWS,
        COLS_N=COLS,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=4,
    )
    return out, out.as_strided(PERMUTE_SHAPE, PERMUTE_STRIDE), sums
