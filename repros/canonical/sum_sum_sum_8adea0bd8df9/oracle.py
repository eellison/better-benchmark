"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ALBERT bf16 tanh-GELU backward and sibling column-reduction scope, including the materialized bf16 `[4096,16384]` GELU-backward tensor, its metadata-only `[16384,4096]` transpose alias, the eleven input `[4096,16384] -> [16384]` fp32 column sums, the GELU-output column sum, the captured f32-sum-to-bf16-to-f32 boundary for every reduction, and the final sequential fp32 add chain. Inductor already schedules this large returned producer plus dependent reductions near the same memory/SFU envelope; a single-pass store-and-reduce plan serializes too much tanh work and recomputing the producer for the reduction would replace one output read with two input reads plus natural `tanh`. The fix is BANDWIDTH_BOUND: keep a full-scope floor oracle and treat any remaining gap as broader pointwise/reduction/memory-traffic codegen work rather than a local algebraic rewrite."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


ROWS = 4096
COLS = 16384
NUM_REDUCTIONS = 12


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
def _gelu_backward_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    half_x = (x * 0.5).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    left = grad * half_x

    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    tanh_val = libdevice.tanh(tanh_arg)
    tanh_plus_one = tanh_val + 1.0

    right = grad * tanh_plus_one
    right_half = (right.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32) * 0.5).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tanh_grad = 1.0 - tanh_val * tanh_val
    d_tanh = (left * tanh_grad) * 0.7978845608028654
    d_tanh_bf16 = d_tanh.to(tl.bfloat16, fp_downcast_rounding="rtne")
    d_cubic = ((d_tanh * 0.044715) * (x2 * 3.0)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    derivative_tail = (d_tanh_bf16.to(tl.float32) + d_cubic.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    out = (derivative_tail.to(tl.float32) + right_half.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _sum_block(matrix, offsets, mask):
    values = tl.load(matrix + offsets, mask=mask, other=0.0).to(tl.float32)
    return tl.sum(tl.where(mask, values, 0.0), axis=0)


@triton.jit
def _all_partials_kernel(
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
    matrix,
    partials,
    N: tl.constexpr,
    C: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    col_vec = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (rows < N) & (cols < C)
    offsets = rows * C + cols

    partial_base = tl.program_id(1) * C + col_vec
    partial_stride = NUM_ROW_BLOCKS * C
    col_mask = col_vec < C

    tl.store(partials + partial_base, _sum_block(matrix0, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride + partial_base, _sum_block(matrix1, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 2 + partial_base, _sum_block(matrix2, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 3 + partial_base, _sum_block(matrix3, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 4 + partial_base, _sum_block(matrix4, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 5 + partial_base, _sum_block(matrix5, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 6 + partial_base, _sum_block(matrix6, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 7 + partial_base, _sum_block(matrix7, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 8 + partial_base, _sum_block(matrix8, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 9 + partial_base, _sum_block(matrix9, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 10 + partial_base, _sum_block(matrix10, offsets, mask), mask=col_mask)
    tl.store(partials + partial_stride * 11 + partial_base, _sum_block(matrix, offsets, mask), mask=col_mask)


@triton.jit
def _finalize_kernel(
    partials,
    out,
    C: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    p = tl.arange(0, BLOCK_P)[:, None]
    mask = (p < NUM_ROW_BLOCKS) & (cols < C)
    offsets = p * C + cols
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


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


# (13 x T([4096,16384], bf16), 11 x S([16384]), S([8,512,16384]), S([8,512,16384]), S([4096,16384]), S([16384]))
@oracle_impl(
    hardware="B200",
    point="e4fe2abf",
    POINTWISE_BLOCK=1024,
    BLOCK_ROWS=512,
    BLOCK_COLS=64,
    FINAL_BLOCK_COLS=16,
    pointwise_warps=8,
    partial_warps=8,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    POINTWISE_BLOCK: int,
    BLOCK_ROWS: int,
    BLOCK_COLS: int,
    FINAL_BLOCK_COLS: int,
    pointwise_warps: int,
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
        arg12,
        *_shape_args,
    ) = inputs

    num_row_blocks = triton.cdiv(ROWS, BLOCK_ROWS)
    block_p = _ceil_pow2(num_row_blocks)
    out = torch.empty_strided((ROWS, COLS), (COLS, 1), device=arg0.device, dtype=torch.bfloat16)
    partials = torch.empty(
        (NUM_REDUCTIONS, num_row_blocks, COLS),
        device=arg0.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty((COLS,), device=arg0.device, dtype=torch.float32)

    _gelu_backward_kernel[(triton.cdiv(ROWS * COLS, POINTWISE_BLOCK),)](
        arg11,
        arg12,
        out,
        N_ELEMENTS=ROWS * COLS,
        BLOCK_SIZE=POINTWISE_BLOCK,
        num_warps=pointwise_warps,
    )

    grid = (triton.cdiv(COLS, BLOCK_COLS), num_row_blocks)
    _all_partials_kernel[grid](
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
        out,
        partials,
        N=ROWS,
        C=COLS,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=partial_warps,
    )

    _finalize_kernel[(triton.cdiv(COLS, FINAL_BLOCK_COLS),)](
        partials,
        sum_out,
        C=COLS,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_P=block_p,
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=final_warps,
    )
    return out, out.permute(1, 0), sum_out
