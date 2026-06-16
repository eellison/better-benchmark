"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 ALBERT attention-score layout plus twelve rounded column-sum scope by materializing the required scaled contiguous `[4096,4096]` clone, returning its aliasing transpose view, and cooperatively split-K reducing the eleven input matrices plus the materialized clone with the explicit f32-sum-to-bf16-to-f32 boundary before the sequential f32 additions, whereas Inductor lowers the independent column reductions, scaled layout clone, alias return, and final vector add chain as generic reduction and pointwise/layout kernels over large intermediates; Inductor cannot do this today because its scheduler/codegen has no multi-input cooperative split-K template that coordinates many same-shape column reductions with a required layout-materializing side output and per-reduction bf16 rounding boundaries; the fix is COOPERATIVE_SPLIT_K: add a guarded ALBERT column-reduction/layout-copy lowering that writes the scaled clone once, emits compact per-column partials for every contributing sum, and finalizes the rounded add chain directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 4096
CHANNELS = 4096
SCALE = 0.3535533905932738


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
def _all_partials_and_layout_kernel(
    matrix0,
    matrix1,
    matrix2,
    matrix3,
    matrix4,
    matrix5,
    matrix6,
    matrix7,
    matrix8,
    matrix9,
    matrix10,
    layout_in,
    clone_out,
    partials,
    N: tl.constexpr,
    C: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    row = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    col_vec = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (row < N) & (col < C)
    offsets = row * C + col
    partial_base = tl.program_id(1) * C + col_vec
    partial_stride = NUM_ROW_BLOCKS * C
    col_mask = col_vec < C

    values = tl.load(matrix0 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix1 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix2 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride * 2 + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix3 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride * 3 + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix4 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride * 4 + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix5 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride * 5 + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix6 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride * 6 + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix7 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride * 7 + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix8 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride * 8 + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix9 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride * 9 + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)
    values = tl.load(matrix10 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials + partial_stride * 10 + partial_base, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=col_mask)

    batch = row // 512
    seq = row - batch * 512
    head = col // 64
    dim = col - head * 64
    layout_offsets = batch * 2097152 + head * 32768 + dim * 512 + seq
    values = tl.load(layout_in + layout_offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = (values * SCALE_).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(clone_out + offsets, scaled, mask=mask)
    sums = tl.sum(tl.where(mask, scaled.to(tl.float32), 0.0), axis=0)
    tl.store(partials + partial_stride * 11 + partial_base, sums, mask=col_mask)


@triton.jit
def _finalize_kernel(
    partials,
    out,
    C: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    p = tl.arange(0, BLOCK_P)[:, None]
    mask = (p < NUM_ROW_BLOCKS) & (col < C)
    offsets = p * C + col
    partial_stride = NUM_ROW_BLOCKS * C

    s0 = tl.sum(tl.load(partials + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s1 = tl.sum(tl.load(partials + partial_stride + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s2 = tl.sum(tl.load(partials + partial_stride * 2 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s3 = tl.sum(tl.load(partials + partial_stride * 3 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s4 = tl.sum(tl.load(partials + partial_stride * 4 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s5 = tl.sum(tl.load(partials + partial_stride * 5 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s6 = tl.sum(tl.load(partials + partial_stride * 6 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s7 = tl.sum(tl.load(partials + partial_stride * 7 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s8 = tl.sum(tl.load(partials + partial_stride * 8 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s9 = tl.sum(tl.load(partials + partial_stride * 9 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s10 = tl.sum(tl.load(partials + partial_stride * 10 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    s11 = tl.sum(tl.load(partials + partial_stride * 11 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    total = _f32_add(s0, s1)
    total = _f32_add(total, s2)
    total = _f32_add(total, s3)
    total = _f32_add(total, s4)
    total = _f32_add(total, s5)
    total = _f32_add(total, s6)
    total = _f32_add(total, s7)
    total = _f32_add(total, s8)
    total = _f32_add(total, s9)
    total = _f32_add(total, s10)
    total = _f32_add(total, s11)

    col_vec = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    tl.store(out + col_vec, total, mask=col_vec < C)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="f2a6d7d1",
    BLOCK_ROWS=128,
    BLOCK_COLS=64,
    FINAL_BLOCK_COLS=16,
    partial_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_ROWS: int,
    BLOCK_COLS: int,
    FINAL_BLOCK_COLS: int,
    partial_warps: int,
    final_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        *_shape_args,
    ) = inputs

    num_row_blocks = triton.cdiv(ROWS, BLOCK_ROWS)
    block_p = _ceil_pow2(num_row_blocks)

    clone = torch.empty_strided((ROWS, CHANNELS), (CHANNELS, 1), device=arg0.device, dtype=torch.bfloat16)
    partials = torch.empty((12, num_row_blocks, CHANNELS), device=arg0.device, dtype=torch.float32)
    out = torch.empty((CHANNELS,), device=arg0.device, dtype=torch.float32)

    grid = (triton.cdiv(CHANNELS, BLOCK_COLS), num_row_blocks)
    _all_partials_and_layout_kernel[grid](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        clone,
        partials,
        N=ROWS,
        C=CHANNELS,
        NUM_ROW_BLOCKS=num_row_blocks,
        SCALE_=SCALE,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=partial_warps,
    )
    _finalize_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_COLS),)](
        partials,
        out,
        C=CHANNELS,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_P=block_p,
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=final_warps,
    )

    return clone, clone.permute(1, 0), out
