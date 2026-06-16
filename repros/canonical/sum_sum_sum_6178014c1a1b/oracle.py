"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ALBERT multi-gradient reduction scope by cooperatively split-K reducing the eleven fp32 product/source column-sum pairs and eleven bf16-rounded matrix column sums, then sharing one row-resident layer-norm-backward producer across the returned bf16 dense tensor, its `[4096,4096]` view and transpose aliases, and three tail column reductions before finalizing the exact sequential add chains. Inductor lowers the independent column reductions, row-local hidden reductions, dense epilogue, bf16 materialization, alias returns, and final vector additions as separate generic reduction and pointwise/layout regions; it cannot currently coordinate this many same-shape reductions with row-local layer-norm scalars, live dense side outputs, alias-only returns, and per-reduction bf16 rounding boundaries in one schedule. The fix is COOPERATIVE_SPLIT_K: add an ALBERT-specific cooperative reduction template that emits compact partials for all compatible column sums, keeps row scalars live through the dense epilogue, writes observable side outputs once, and finalizes rounded add chains directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 4096
CHANNELS = 4096
ROW_FACTOR = 4096.0
INV_ROW_FACTOR = 1.0 / 4096.0


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pair_partial_kernel(
    lhs_ptr,
    rhs_ptr,
    prod_partial_ptr,
    src_partial_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    col_vec = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (rows < ROWS_) & (cols < CHANNELS_)
    offsets = rows * CHANNELS_ + cols

    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    prod = _mul_rn(lhs, rhs)

    partial_base = tl.program_id(1) * CHANNELS_ + col_vec
    col_mask = col_vec < CHANNELS_
    tl.store(
        prod_partial_ptr + partial_base,
        tl.sum(tl.where(mask, prod, 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        src_partial_ptr + partial_base,
        tl.sum(tl.where(mask, lhs, 0.0), axis=0),
        mask=col_mask,
    )


@triton.jit
def _sum_matrix_block(matrix_ptr, offsets, mask):
    values = tl.load(matrix_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    return tl.sum(tl.where(mask, values, 0.0), axis=0)


@triton.jit
def _all_bf16_matrix_partials_kernel(
    matrix0_ptr,
    matrix1_ptr,
    matrix2_ptr,
    matrix3_ptr,
    matrix4_ptr,
    matrix5_ptr,
    matrix6_ptr,
    matrix7_ptr,
    matrix8_ptr,
    matrix9_ptr,
    matrix10_ptr,
    partials_ptr,
    NUM_MAIN_GROUPS: tl.constexpr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    col_vec = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (rows < ROWS_) & (cols < CHANNELS_)
    offsets = rows * CHANNELS_ + cols

    partial_base = tl.program_id(1) * CHANNELS_ + col_vec
    partial_stride = NUM_MAIN_GROUPS * CHANNELS_
    col_mask = col_vec < CHANNELS_

    tl.store(partials_ptr + partial_base, _sum_matrix_block(matrix0_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride + partial_base, _sum_matrix_block(matrix1_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride * 2 + partial_base, _sum_matrix_block(matrix2_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride * 3 + partial_base, _sum_matrix_block(matrix3_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride * 4 + partial_base, _sum_matrix_block(matrix4_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride * 5 + partial_base, _sum_matrix_block(matrix5_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride * 6 + partial_base, _sum_matrix_block(matrix6_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride * 7 + partial_base, _sum_matrix_block(matrix7_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride * 8 + partial_base, _sum_matrix_block(matrix8_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride * 9 + partial_base, _sum_matrix_block(matrix9_ptr, offsets, mask), mask=col_mask)
    tl.store(partials_ptr + partial_stride * 10 + partial_base, _sum_matrix_block(matrix10_ptr, offsets, mask), mask=col_mask)


@triton.jit
def _tail_row_partials_kernel(
    arg33_ptr,
    arg34_ptr,
    weight_ptr,
    norm_source_ptr,
    row_center_ptr,
    row_scale_ptr,
    bf16_out_ptr,
    tail_partials_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    INV_ROW_FACTOR_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    row_factor = tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32)

    acc_prod = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_src = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_bf16 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS_
        offsets = rows[:, None] * CHANNELS_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        src = _add_rn(
            tl.load(arg34_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            tl.load(arg33_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        )
        norm_source = tl.load(norm_source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        center = tl.load(row_center_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        normed = _mul_rn(_sub_rn(norm_source, center[:, None]), scale[:, None])

        weighted = _mul_rn(src, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, normed), 0.0), axis=1)
        centered = _sub_rn(_mul_rn(weighted, row_factor), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(normed, row_dot[:, None]))
        row_factor_scale = _mul_rn(scale, INV_ROW_FACTOR_)
        grad = _mul_rn(row_factor_scale[:, None], centered)
        grad_bf16 = grad.to(tl.bfloat16)

        tl.store(bf16_out_ptr + offsets, grad_bf16, mask=mask)

        acc_prod += tl.sum(tl.where(mask, _mul_rn(src, normed), 0.0), axis=0)
        acc_src += tl.sum(tl.where(mask, src, 0.0), axis=0)
        acc_bf16 += tl.sum(tl.where(mask, grad_bf16.to(tl.float32), 0.0), axis=0)

    partial_base = group * 3 * CHANNELS_ + cols
    tl.store(tail_partials_ptr + partial_base, acc_prod, mask=col_mask)
    tl.store(tail_partials_ptr + partial_base + CHANNELS_, acc_src, mask=col_mask)
    tl.store(tail_partials_ptr + partial_base + 2 * CHANNELS_, acc_bf16, mask=col_mask)


@triton.jit
def _finalize_all_kernel(
    pair_partials_ptr,
    bf16_partials_ptr,
    tail_partials_ptr,
    prod_out_ptr,
    src_out_ptr,
    bf16_sum_out_ptr,
    NUM_MAIN_GROUPS: tl.constexpr,
    NUM_TAIL_GROUPS: tl.constexpr,
    CHANNELS_: tl.constexpr,
    MAIN_GROUP_BLOCK: tl.constexpr,
    TAIL_GROUP_BLOCK: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = cols < CHANNELS_

    main_groups = tl.arange(0, MAIN_GROUP_BLOCK)[:, None]
    main_offsets = main_groups * CHANNELS_ + cols[None, :]
    main_mask = (main_groups < NUM_MAIN_GROUPS) & col_mask[None, :]
    main_stride = NUM_MAIN_GROUPS * CHANNELS_

    p0 = tl.sum(tl.load(pair_partials_ptr + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p1 = tl.sum(tl.load(pair_partials_ptr + main_stride + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p2 = tl.sum(tl.load(pair_partials_ptr + main_stride * 2 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p3 = tl.sum(tl.load(pair_partials_ptr + main_stride * 3 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p4 = tl.sum(tl.load(pair_partials_ptr + main_stride * 4 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p5 = tl.sum(tl.load(pair_partials_ptr + main_stride * 5 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p6 = tl.sum(tl.load(pair_partials_ptr + main_stride * 6 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p7 = tl.sum(tl.load(pair_partials_ptr + main_stride * 7 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p8 = tl.sum(tl.load(pair_partials_ptr + main_stride * 8 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p9 = tl.sum(tl.load(pair_partials_ptr + main_stride * 9 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    p10 = tl.sum(tl.load(pair_partials_ptr + main_stride * 20 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)

    s0 = tl.sum(tl.load(pair_partials_ptr + main_stride * 10 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s1 = tl.sum(tl.load(pair_partials_ptr + main_stride * 11 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s2 = tl.sum(tl.load(pair_partials_ptr + main_stride * 12 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s3 = tl.sum(tl.load(pair_partials_ptr + main_stride * 13 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s4 = tl.sum(tl.load(pair_partials_ptr + main_stride * 14 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s5 = tl.sum(tl.load(pair_partials_ptr + main_stride * 15 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s6 = tl.sum(tl.load(pair_partials_ptr + main_stride * 16 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s7 = tl.sum(tl.load(pair_partials_ptr + main_stride * 17 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s8 = tl.sum(tl.load(pair_partials_ptr + main_stride * 18 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s9 = tl.sum(tl.load(pair_partials_ptr + main_stride * 19 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)
    s10 = tl.sum(tl.load(pair_partials_ptr + main_stride * 21 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0)

    b0 = tl.sum(tl.load(bf16_partials_ptr + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b1 = tl.sum(tl.load(bf16_partials_ptr + main_stride + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b2 = tl.sum(tl.load(bf16_partials_ptr + main_stride * 2 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b3 = tl.sum(tl.load(bf16_partials_ptr + main_stride * 3 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b4 = tl.sum(tl.load(bf16_partials_ptr + main_stride * 4 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b5 = tl.sum(tl.load(bf16_partials_ptr + main_stride * 5 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b6 = tl.sum(tl.load(bf16_partials_ptr + main_stride * 6 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b7 = tl.sum(tl.load(bf16_partials_ptr + main_stride * 7 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b8 = tl.sum(tl.load(bf16_partials_ptr + main_stride * 8 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b9 = tl.sum(tl.load(bf16_partials_ptr + main_stride * 9 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    b10 = tl.sum(tl.load(bf16_partials_ptr + main_stride * 10 + main_offsets, mask=main_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)

    tail_groups = tl.arange(0, TAIL_GROUP_BLOCK)[:, None]
    tail_offsets = tail_groups * 3 * CHANNELS_ + cols[None, :]
    tail_mask = (tail_groups < NUM_TAIL_GROUPS) & col_mask[None, :]
    tail_prod = tl.sum(tl.load(tail_partials_ptr + tail_offsets, mask=tail_mask, other=0.0).to(tl.float32), axis=0)
    tail_src = tl.sum(tl.load(tail_partials_ptr + tail_offsets + CHANNELS_, mask=tail_mask, other=0.0).to(tl.float32), axis=0)
    tail_bf16 = tl.sum(tl.load(tail_partials_ptr + tail_offsets + 2 * CHANNELS_, mask=tail_mask, other=0.0).to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)

    prod_total = _add_rn(p0, p1)
    prod_total = _add_rn(prod_total, p2)
    prod_total = _add_rn(prod_total, p3)
    prod_total = _add_rn(prod_total, p4)
    prod_total = _add_rn(prod_total, p5)
    prod_total = _add_rn(prod_total, p6)
    prod_total = _add_rn(prod_total, p7)
    prod_total = _add_rn(prod_total, p8)
    prod_total = _add_rn(prod_total, p9)
    prod_total = _add_rn(prod_total, p10)
    prod_total = _add_rn(prod_total, tail_prod)

    src_total = _add_rn(s0, s1)
    src_total = _add_rn(src_total, s2)
    src_total = _add_rn(src_total, s3)
    src_total = _add_rn(src_total, s4)
    src_total = _add_rn(src_total, s5)
    src_total = _add_rn(src_total, s6)
    src_total = _add_rn(src_total, s7)
    src_total = _add_rn(src_total, s8)
    src_total = _add_rn(src_total, s9)
    src_total = _add_rn(src_total, s10)
    src_total = _add_rn(src_total, tail_src)

    bf16_total = _add_rn(b0, b1)
    bf16_total = _add_rn(bf16_total, b2)
    bf16_total = _add_rn(bf16_total, b3)
    bf16_total = _add_rn(bf16_total, b4)
    bf16_total = _add_rn(bf16_total, b5)
    bf16_total = _add_rn(bf16_total, b6)
    bf16_total = _add_rn(bf16_total, b7)
    bf16_total = _add_rn(bf16_total, b8)
    bf16_total = _add_rn(bf16_total, b9)
    bf16_total = _add_rn(bf16_total, b10)
    bf16_total = _add_rn(bf16_total, tail_bf16)

    tl.store(prod_out_ptr + cols, prod_total, mask=col_mask)
    tl.store(src_out_ptr + cols, src_total, mask=col_mask)
    tl.store(bf16_sum_out_ptr + cols, bf16_total, mask=col_mask)


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


# (33 reduction inputs plus ALBERT layernorm-backward tail, shape hash 99a4c701)
@oracle_impl(
    hardware="B200",
    point="99a4c701",
    MAIN_BLOCK_ROWS=128,
    MAIN_BLOCK_COLS=64,
    TAIL_ROWS_PER_GROUP=4,
    TAIL_BLOCK_R=1,
    TAIL_BLOCK_C=4096,
    FINAL_BLOCK_COLS=8,
    main_warps=8,
    tail_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    MAIN_BLOCK_ROWS: int,
    MAIN_BLOCK_COLS: int,
    TAIL_ROWS_PER_GROUP: int,
    TAIL_BLOCK_R: int,
    TAIL_BLOCK_C: int,
    FINAL_BLOCK_COLS: int,
    main_warps: int,
    tail_warps: int,
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
        arg13,
        arg14,
        arg15,
        arg16,
        arg17,
        arg18,
        arg19,
        arg20,
        arg21,
        arg22,
        arg23,
        arg24,
        arg25,
        arg26,
        arg27,
        arg28,
        arg29,
        arg30,
        arg31,
        arg32,
        arg33,
        arg34,
        arg35,
        arg36,
        arg37,
        arg38,
        *_shape_params,
    ) = inputs

    device = arg0.device
    rows = ROWS
    channels = CHANNELS
    main_groups = triton.cdiv(rows, MAIN_BLOCK_ROWS)
    tail_groups = triton.cdiv(rows, TAIL_ROWS_PER_GROUP)

    pair_partials = torch.empty((22, main_groups, channels), device=device, dtype=torch.float32)
    bf16_partials = torch.empty((11, main_groups, channels), device=device, dtype=torch.float32)
    tail_partials = torch.empty_strided(
        (tail_groups, 3, channels),
        (3 * channels, channels, 1),
        device=device,
        dtype=torch.float32,
    )

    bf16_3d = torch.empty_strided(
        tuple(int(dim) for dim in arg34.shape),
        tuple(int(stride) for stride in arg34.stride()),
        device=device,
        dtype=torch.bfloat16,
    )
    prod_out = torch.empty((channels,), device=device, dtype=torch.float32)
    src_out = torch.empty((channels,), device=device, dtype=torch.float32)
    bf16_sum_out = torch.empty((channels,), device=device, dtype=torch.float32)

    main_grid = (triton.cdiv(channels, MAIN_BLOCK_COLS), main_groups)
    _pair_partial_kernel[main_grid](arg0, arg1, pair_partials[0], pair_partials[10], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg3, arg4, pair_partials[1], pair_partials[11], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg6, arg7, pair_partials[2], pair_partials[12], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg9, arg10, pair_partials[3], pair_partials[13], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg12, arg13, pair_partials[4], pair_partials[14], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg15, arg16, pair_partials[5], pair_partials[15], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg18, arg19, pair_partials[6], pair_partials[16], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg21, arg22, pair_partials[7], pair_partials[17], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg24, arg25, pair_partials[8], pair_partials[18], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg27, arg28, pair_partials[9], pair_partials[19], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)
    _pair_partial_kernel[main_grid](arg30, arg31, pair_partials[20], pair_partials[21], ROWS_=rows, CHANNELS_=channels, BLOCK_ROWS=MAIN_BLOCK_ROWS, BLOCK_COLS=MAIN_BLOCK_COLS, num_warps=main_warps)

    _all_bf16_matrix_partials_kernel[main_grid](
        arg2,
        arg5,
        arg8,
        arg11,
        arg14,
        arg17,
        arg20,
        arg23,
        arg26,
        arg29,
        arg32,
        bf16_partials,
        NUM_MAIN_GROUPS=main_groups,
        ROWS_=rows,
        CHANNELS_=channels,
        BLOCK_ROWS=MAIN_BLOCK_ROWS,
        BLOCK_COLS=MAIN_BLOCK_COLS,
        num_warps=main_warps,
    )

    _tail_row_partials_kernel[(tail_groups,)](
        arg33,
        arg34,
        arg35,
        arg36,
        arg37,
        arg38,
        bf16_3d,
        tail_partials,
        ROWS_=rows,
        CHANNELS_=channels,
        ROWS_PER_GROUP=TAIL_ROWS_PER_GROUP,
        BLOCK_R=TAIL_BLOCK_R,
        BLOCK_C=TAIL_BLOCK_C,
        ROW_FACTOR_=ROW_FACTOR,
        INV_ROW_FACTOR_=INV_ROW_FACTOR,
        num_warps=tail_warps,
    )

    _finalize_all_kernel[(triton.cdiv(channels, FINAL_BLOCK_COLS),)](
        pair_partials,
        bf16_partials,
        tail_partials,
        prod_out,
        src_out,
        bf16_sum_out,
        NUM_MAIN_GROUPS=main_groups,
        NUM_TAIL_GROUPS=tail_groups,
        CHANNELS_=channels,
        MAIN_GROUP_BLOCK=_ceil_pow2(main_groups),
        TAIL_GROUP_BLOCK=_ceil_pow2(tail_groups),
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=final_warps,
    )

    bf16_2d = bf16_3d.view(rows, channels)
    return prod_out, src_out, bf16_3d, bf16_2d, bf16_2d.permute(1, 0), bf16_sum_out
