"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 ALBERT attention-score transpose plus twelve rounded column-sum scope by materializing the required contiguous `[4096, 4096]` clone, returning its aliasing transpose view, and cooperatively split-K reducing the eleven input matrices plus the materialized clone with the explicit f32-sum-to-bf16-to-f32 boundary before the sequential f32 additions, whereas Inductor lowers the independent column reductions, layout clone, alias return, and final vector add chain as generic reduction and pointwise/layout kernels over large intermediates; Inductor cannot do this today because its scheduler/codegen has no multi-input cooperative split-K template that coordinates many same-shape column reductions with a required layout-materializing side output and per-reduction bf16 rounding boundaries; the fix is COOPERATIVE_SPLIT_K: add a guarded ALBERT column-reduction/layout-copy lowering that writes the clone once, emits compact per-column partials for every contributing sum, and finalizes the rounded add chain directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _matrix_partial_kernel(
    matrix,
    partial,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    row = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    col_vec = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (row < N) & (col < C)
    offsets = row * C + col

    partial_base = tl.program_id(1) * C + col_vec
    col_mask = col_vec < C

    values = tl.load(matrix + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(partial + partial_base, sums, mask=col_mask)


@triton.jit
def _layout_partial_kernel(
    layout_in,
    clone_out,
    partial,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    row = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    col_vec = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (row < N) & (col < C)
    offsets = row * C + col
    partial_base = tl.program_id(1) * C + col_vec
    col_mask = col_vec < C

    batch = row // 512
    seq = row - batch * 512
    head = col // 64
    dim = col - head * 64
    layout_offsets = batch * 2097152 + head * 32768 + seq * 64 + dim
    values = tl.load(layout_in + layout_offsets, mask=mask, other=0.0)
    tl.store(clone_out + offsets, values, mask=mask)
    sums = tl.sum(tl.where(mask, values.to(tl.float32), 0.0), axis=0)
    tl.store(partial + partial_base, sums, mask=col_mask)


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

    s0 = tl.sum(tl.load(partials + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s1 = tl.sum(tl.load(partials + partial_stride + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s2 = tl.sum(tl.load(partials + partial_stride * 2 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s3 = tl.sum(tl.load(partials + partial_stride * 3 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s4 = tl.sum(tl.load(partials + partial_stride * 4 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s5 = tl.sum(tl.load(partials + partial_stride * 5 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s6 = tl.sum(tl.load(partials + partial_stride * 6 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s7 = tl.sum(tl.load(partials + partial_stride * 7 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s8 = tl.sum(tl.load(partials + partial_stride * 8 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s9 = tl.sum(tl.load(partials + partial_stride * 9 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s10 = tl.sum(tl.load(partials + partial_stride * 10 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    s11 = tl.sum(tl.load(partials + partial_stride * 11 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)

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
    point="4a97732f",
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
    n = 4096
    c = 4096
    num_row_blocks = triton.cdiv(n, BLOCK_ROWS)
    block_p = _ceil_pow2(num_row_blocks)

    clone = torch.empty_strided((n, c), (c, 1), device=arg0.device, dtype=torch.bfloat16)
    partials = torch.empty((12, num_row_blocks, c), device=arg0.device, dtype=torch.float32)
    out = torch.empty((c,), device=arg0.device, dtype=torch.float32)

    grid = (triton.cdiv(c, BLOCK_COLS), num_row_blocks)
    _matrix_partial_kernel[grid](arg0, partials[0], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg1, partials[1], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg2, partials[2], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg3, partials[3], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg4, partials[4], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg5, partials[5], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg6, partials[6], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg7, partials[7], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg8, partials[8], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg9, partials[9], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _matrix_partial_kernel[grid](arg10, partials[10], N=n, C=c, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, num_warps=partial_warps)
    _layout_partial_kernel[grid](
        arg11,
        clone,
        partials[11],
        N=n,
        C=c,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=partial_warps,
    )
    _finalize_kernel[(triton.cdiv(c, FINAL_BLOCK_COLS),)](
        partials,
        out,
        C=c,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_P=block_p,
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=final_warps,
    )
    return clone, clone.permute(1, 0), out
