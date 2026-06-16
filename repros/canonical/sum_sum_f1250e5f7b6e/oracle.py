"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete T5 bf16 residual/dropout/layer-norm-backward scope by sharing the twelve-bf16-plus-f32 producer across the returned `[512]` column sum, the row-local `[2]` reduction that feeds the f32 dense output, and the final bf16 dropout dense output plus its transpose view, whereas Inductor materializes the shared dropout producer and schedules the sibling column reduction as separate reduction/finalization kernels after the row-reduction epilogue; Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K multi-output template that coordinates row partials, compatible column accumulators, full dense f32/bf16 stores, and the aliasing transpose view while preserving the captured f32 operation order; the fix is COOPERATIVE_SPLIT_K: add a guarded T5 backward plan that emits row partials and column sums from one producer pass, co-finalizes the row epilogue, and writes both dense outputs in their destination layouts."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 8192
COLS = 512
MASK_SCALE = 1.1111111111111112
INV_COLS = 0.001953125


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
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
def _producer_sum(
    residual0_ptr,
    base_ptr,
    residual1_ptr,
    residual2_ptr,
    residual3_ptr,
    residual4_ptr,
    residual5_ptr,
    residual6_ptr,
    residual7_ptr,
    residual8_ptr,
    residual9_ptr,
    residual10_ptr,
    residual11_ptr,
    offsets,
    active,
):
    acc = tl.load(base_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    acc = _f32_add(acc, tl.load(residual0_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual1_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual2_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual3_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual4_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual5_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual6_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual7_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual8_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual9_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual10_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    acc = _f32_add(acc, tl.load(residual11_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    return acc


@triton.jit
def _zero_kernel(out_ptr, COLS_: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + cols, tl.zeros((BLOCK,), tl.float32), mask=cols < COLS_)


@triton.jit
def _partials_kernel(
    residual0_ptr,
    base_ptr,
    residual1_ptr,
    residual2_ptr,
    residual3_ptr,
    residual4_ptr,
    residual5_ptr,
    residual6_ptr,
    residual7_ptr,
    residual8_ptr,
    residual9_ptr,
    residual10_ptr,
    residual11_ptr,
    mask1_ptr,
    weight_ptr,
    src_ptr,
    row_scale_ptr,
    row_partial_ptr,
    col_sum_ptr,
    weighted_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    MASK_SCALE_: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_block * COL_BLOCK + tl.arange(0, COL_BLOCK)
    offsets = rows[:, None] * COLS_ + cols[None, :]
    active = (rows[:, None] < ROWS_) & (cols[None, :] < COLS_)

    total = _producer_sum(
        residual0_ptr,
        base_ptr,
        residual1_ptr,
        residual2_ptr,
        residual3_ptr,
        residual4_ptr,
        residual5_ptr,
        residual6_ptr,
        residual7_ptr,
        residual8_ptr,
        residual9_ptr,
        residual10_ptr,
        residual11_ptr,
        offsets,
        active,
    )
    mask1 = tl.load(mask1_ptr + offsets, mask=active, other=0).to(tl.float32)
    dropped = _f32_mul(total, _f32_mul(mask1, MASK_SCALE_))
    weight = tl.load(weight_ptr + cols, mask=cols < COLS_, other=0.0).to(tl.float32)
    weighted = _f32_mul(dropped, weight[None, :])
    src = tl.load(src_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + rows, mask=rows < ROWS_, other=0.0).to(tl.float32)

    row_terms = _f32_mul(weighted, src)
    src_scale = _f32_mul(src, row_scale[:, None])
    col_terms = _f32_mul(dropped, src_scale)

    row_sums = tl.sum(tl.where(active, row_terms, 0.0), axis=1)
    col_sums = tl.sum(tl.where(active, col_terms, 0.0), axis=0)

    tl.store(weighted_ptr + offsets, weighted, mask=active)
    tl.store(row_partial_ptr + col_block * ROWS_ + rows, row_sums, mask=rows < ROWS_)
    tl.atomic_add(col_sum_ptr + cols, col_sums, sem="relaxed", mask=cols < COLS_)


@triton.jit
def _epilogue_kernel(
    weighted_ptr,
    src_ptr,
    row_scale_ptr,
    mask2_ptr,
    row_partial_ptr,
    dense_f32_ptr,
    dense_bf16_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    MASK_SCALE_: tl.constexpr,
    INV_COLS_: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    COL_BLOCKS: tl.constexpr,
    BLOCK_CB: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_block * COL_BLOCK + tl.arange(0, COL_BLOCK)
    offsets = rows[:, None] * COLS_ + cols[None, :]
    active = (rows[:, None] < ROWS_) & (cols[None, :] < COLS_)

    weighted = tl.load(weighted_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    src = tl.load(src_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + rows, mask=rows < ROWS_, other=0.0).to(tl.float32)

    blocks = tl.arange(0, BLOCK_CB)
    row_values = tl.load(
        row_partial_ptr + blocks[:, None] * ROWS_ + rows[None, :],
        mask=(blocks[:, None] < COL_BLOCKS) & (rows[None, :] < ROWS_),
        other=0.0,
    ).to(tl.float32)
    row_total = tl.sum(row_values, axis=0)

    direct = _f32_mul(weighted, row_scale[:, None])
    row_scale_sq = _f32_mul(row_scale, row_scale)
    row_scale_cu = _f32_mul(row_scale_sq, row_scale)
    correction0 = _f32_mul(row_total, -0.5)
    correction1 = _f32_mul(correction0, row_scale_cu)
    correction2 = _f32_mul(correction1, INV_COLS_)
    correction3 = _f32_mul(src, 2.0)
    correction = _f32_mul(correction2[:, None], correction3)
    dense_f32 = _f32_add(direct, correction)

    mask2 = tl.load(mask2_ptr + offsets, mask=active, other=0).to(tl.float32)
    dense_bf16 = _f32_mul(dense_f32, _f32_mul(mask2, MASK_SCALE_))

    tl.store(dense_f32_ptr + offsets, dense_f32, mask=active)
    tl.store(dense_bf16_ptr + offsets, dense_bf16, mask=active)


@oracle_impl(
    hardware="B200",
    point="6927d79e",
    ROW_BLOCK=16,
    COL_BLOCK=64,
    zero_warps=4,
    partial_warps=4,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    COL_BLOCK: int,
    zero_warps: int,
    partial_warps: int,
    epilogue_warps: int,
):
    (
        residual0,
        base,
        residual1,
        residual2,
        residual3,
        residual4,
        residual5,
        residual6,
        residual7,
        residual8,
        residual9,
        residual10,
        residual11,
        mask1,
        weight,
        src,
        row_scale,
        mask2,
        *_shape_params,
    ) = inputs

    device = base.device
    row_blocks = triton.cdiv(ROWS, ROW_BLOCK)
    col_blocks = triton.cdiv(COLS, COL_BLOCK)
    row_partial = torch.empty((col_blocks, ROWS), device=device, dtype=torch.float32)
    weighted = torch.empty_strided((ROWS, COLS), (COLS, 1), device=device, dtype=torch.float32)
    col_sum = torch.empty_strided((COLS,), (1,), device=device, dtype=torch.float32)
    dense_f32 = torch.empty_strided(
        (8, 1024, COLS),
        (1024 * COLS, COLS, 1),
        device=device,
        dtype=torch.float32,
    )
    dense_bf16 = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _zero_kernel[(triton.cdiv(COLS, 256),)](
        col_sum,
        COLS_=COLS,
        BLOCK=256,
        num_warps=zero_warps,
        num_stages=3,
    )
    _partials_kernel[(row_blocks, col_blocks)](
        residual0,
        base,
        residual1,
        residual2,
        residual3,
        residual4,
        residual5,
        residual6,
        residual7,
        residual8,
        residual9,
        residual10,
        residual11,
        mask1,
        weight,
        src,
        row_scale,
        row_partial,
        col_sum,
        weighted,
        ROWS_=ROWS,
        COLS_=COLS,
        MASK_SCALE_=MASK_SCALE,
        ROW_BLOCK=ROW_BLOCK,
        COL_BLOCK=COL_BLOCK,
        num_warps=partial_warps,
        num_stages=3,
    )
    _epilogue_kernel[(row_blocks, col_blocks)](
        weighted,
        src,
        row_scale,
        mask2,
        row_partial,
        dense_f32,
        dense_bf16,
        ROWS_=ROWS,
        COLS_=COLS,
        MASK_SCALE_=MASK_SCALE,
        INV_COLS_=INV_COLS,
        ROW_BLOCK=ROW_BLOCK,
        COL_BLOCK=COL_BLOCK,
        COL_BLOCKS=col_blocks,
        BLOCK_CB=8,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return col_sum, dense_f32, dense_bf16, torch.ops.aten.permute.default(dense_bf16, [1, 0])
