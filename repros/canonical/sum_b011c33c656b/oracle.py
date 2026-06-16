"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete bf16 Longformer sliding-window attention-backward scatter/reduction scope by fusing the padded diagonal score assembly, dropout/key/global masks, exact natural-exp probability producer, original-layout row sum, eager-equivalent f32 mul/add, bf16 cast boundaries, boundary slice_scatter patches, and final diagonal output layout into Triton kernels, whereas Inductor lowers the decomposed view/pad/slice_scatter/where/exp/div/sum/fma/select_scatter pipeline through generic materialized layout and reduction kernels; Inductor cannot do this today because its scheduler/codegen has no structured Longformer diagonal scatter-reduce lowering that combines row-local reductions with the surrounding sliding-window scatter layout transforms; the fix is SCATTER_REDUCE: add a Longformer sliding-window backward lowering that preserves bf16 rounding boundaries while writing the final diagonalized output layout directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


HEADS = 12
TOKENS = 1024
CHUNK = 256
OVERLAP_BLOCKS = 4
SOURCE_K = 768
REDUCE_K = 513
OUT_CHANNELS = 3
OUT_ROWS = 512
OUT_COLS = 512
OUT_PLANE = OUT_ROWS * OUT_COLS
TOTAL_OUT = 288 * OUT_PLANE
TOTAL_A2_ROWS = 8 * TOKENS * HEADS
BLOCK_R = 1024
BLOCK_OUT = 256
DROPOUT_SCALE_BF16 = 1.109375


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bf16_add_f32(a, b):
    return (a.to(tl.bfloat16) + b.to(tl.bfloat16)).to(tl.float32)


@triton.jit
def _product_probs_kernel(
    arg0_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    arg7_ptr,
    arg8_ptr,
    arg9_ptr,
    product_ptr,
    probs_ptr,
    BLOCK_R_: tl.constexpr,
):
    row_id = tl.program_id(0)
    r = tl.arange(0, BLOCK_R_)
    valid = r < 513

    h = row_id % 12
    q = row_id // 12
    s = q % 1024
    b = q // 1024
    a = b * 12 + h
    j = s // 256
    t = s - j * 256

    source_offsets = ((a * 4 + j) * 256 + t) * 768 + (t + r)
    key_offsets = b * 6422528 + s * 6272 + h * 513 + r
    prob_offsets = b * 6303744 + s * 513 + h * 525312 + r
    bias_offsets = b * 525312 + s * 513 + r
    stats_offset = ((b * 1024 + s) * 12 + h)
    out_offsets = ((b * 1024 + s) * 12 + h) * 513 + r

    source = tl.load(arg0_ptr + source_offsets, mask=valid, other=0.0).to(tl.float32)
    keep = tl.load(arg3_ptr + key_offsets, mask=valid, other=0) != 0
    scaled_source = _round_to_bf16_f32(source * 1.109375)
    masked_scores = tl.where(keep, scaled_source, 0.0)
    global_mask = tl.load(arg4_ptr + b * 1024 + s) != 0
    fill = tl.load(arg5_ptr).to(tl.float32)
    where_value = tl.where(global_mask, fill, masked_scores)

    logits = tl.load(arg6_ptr + prob_offsets, mask=valid, other=0.0).to(tl.float32)
    bias = tl.load(arg7_ptr + bias_offsets, mask=valid, other=0.0).to(tl.float32)
    added = _round_to_bf16_f32(logits + bias)
    center = tl.load(arg8_ptr + stats_offset).to(tl.float32)
    denom = tl.load(arg9_ptr + stats_offset).to(tl.float32)
    probs = _f32_div(libdevice.exp(added - center), denom)
    product = where_value * probs

    tl.store(product_ptr + out_offsets, product, mask=valid)
    tl.store(probs_ptr + out_offsets, probs, mask=valid)


@triton.jit
def _a2_from_product_kernel(
    product_ptr,
    probs_ptr,
    row_sums_ptr,
    arg10_ptr,
    arg11_ptr,
    arg12_ptr,
    arg13_ptr,
    arg14_ptr,
    arg15_ptr,
    a2_ptr,
    BLOCK_R_: tl.constexpr,
):
    row_id = tl.program_id(0)
    r = tl.arange(0, BLOCK_R_)
    valid = r < 513

    h = row_id % 12
    q = row_id // 12
    s = q % 1024
    b = q // 1024
    a = b * 12 + h
    j = s // 256
    t = s - j * 256

    row_offset = ((a * 4 + j) * 256 + t)
    source_row = ((b * 1024 + s) * 12 + h)
    source_offsets = source_row * 513 + r
    offsets = row_offset * 513 + r
    short_offsets = ((b * 256 + t) * 12 + h) * 257 + r
    long_offsets = ((b * 256 + t) * 12 + h) * 513 + r
    full_offsets = ((b * 1024 + s) * 12 + h) * 513 + r

    product = tl.load(product_ptr + source_offsets, mask=valid, other=0.0).to(tl.float32)
    probs = tl.load(probs_ptr + source_offsets, mask=valid, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sums_ptr + source_row).to(tl.float32)
    grad = _f32_add(_f32_mul(-probs, row_sum), product)
    g = _round_to_bf16_f32(grad)

    arg14 = tl.load(arg14_ptr + full_offsets, mask=valid, other=0.0).to(tl.float32)
    arg13 = tl.load(arg13_ptr + long_offsets, mask=valid, other=0.0).to(tl.float32)

    end_u = r - 256
    end_offsets = ((b * 256 + t) * 12 + h) * 257 + end_u
    arg10_end = tl.load(arg10_ptr + end_offsets, mask=valid & (r >= 256), other=0.0).to(tl.float32)
    end_mask = tl.load(arg11_ptr + end_offsets, mask=valid & (r >= 256), other=0) != 0
    scalar_bf16 = tl.load(arg12_ptr).to(tl.float32)

    a1_regular = _bf16_add_f32(g, arg14)
    a1_j3_low = _bf16_add_f32(g, arg13)
    a1_j3_high = _bf16_add_f32(arg10_end, tl.where(end_mask, scalar_bf16, g))
    a1_j3 = tl.where(r >= 256, a1_j3_high, a1_j3_low)
    a1 = tl.where(j == 3, a1_j3, a1_regular)

    start_arg10 = tl.load(arg10_ptr + short_offsets, mask=valid & (r <= 256), other=0.0).to(tl.float32)
    start_mask = tl.load(arg15_ptr + short_offsets, mask=valid & (r <= 256), other=0) != 0
    a2_j0_low = _bf16_add_f32(start_arg10, tl.where(start_mask, scalar_bf16, a1))
    a2_j0_high = _bf16_add_f32(a1, arg13)
    a2_j0 = tl.where(r <= 256, a2_j0_low, a2_j0_high)
    a2_nonzero = _bf16_add_f32(a1, arg14)
    a2 = tl.where(j == 0, a2_j0, a2_nonzero)

    tl.store(a2_ptr + offsets, a2.to(tl.bfloat16), mask=valid)


@triton.jit
def _a2_value(a2_ptr, a, j, t, r, valid):
    offsets = ((a * 4 + j) * 256 + t) * 513 + r
    return tl.load(a2_ptr + offsets, mask=valid, other=0.0).to(tl.float32)


@triton.jit
def _b1_value(a2_ptr, arg16_ptr, a, j, t, r, valid):
    a2 = _a2_value(a2_ptr, a, j, t, r, valid)
    replace = (j == 0) & (t > 0) & (r > 0) & (r < 256)
    full7 = tl.load(
        arg16_ptr + (a * 255 + (t - 1)) * 255 + (r - 1),
        mask=valid & replace,
        other=0.0,
    ).to(tl.float32)
    return tl.where(replace, full7, a2)


@triton.jit
def _b2_value(a2_ptr, arg16_ptr, arg20_ptr, a, j, t, r, valid):
    b1 = _b1_value(a2_ptr, arg16_ptr, a, j, t, r, valid)
    replace = (j >= 1) & (r < 256)
    full11 = tl.load(
        arg20_ptr + (((a * 3 + (j - 1)) * 256 + t) * 256 + r),
        mask=valid & replace,
        other=0.0,
    ).to(tl.float32)
    return tl.where(replace, full11, b1)


@triton.jit
def _b3_value(a2_ptr, arg16_ptr, arg20_ptr, arg22_ptr, a, j, t, r, valid):
    b2 = _b2_value(a2_ptr, arg16_ptr, arg20_ptr, a, j, t, r, valid)
    replace = (j == 3) & (r >= 256)
    full13 = tl.load(
        arg22_ptr + (a * 256 + t) * 257 + (r - 256),
        mask=valid & replace,
        other=0.0,
    ).to(tl.float32)
    return tl.where(replace, full13, b2)


@triton.jit
def _final_output_kernel(
    a2_ptr,
    arg16_ptr,
    arg17_ptr,
    arg18_ptr,
    arg19_ptr,
    arg20_ptr,
    arg21_ptr,
    arg22_ptr,
    arg23_ptr,
    out_ptr,
    TOTAL_: tl.constexpr,
    BLOCK_: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
    valid = offsets < TOTAL_

    p = offsets // 262144
    rem = offsets - p * 262144
    v = rem // 512
    w = rem - v * 512

    add_linear = (p * 513 + v) * 512 + w
    q = add_linear // 513
    col = add_linear - q * 513
    row = q % 512
    pc = q // 512
    c = pc % 3
    a = pc // 3

    full10 = tl.load(
        arg19_ptr + (((a * 3 + c) * 512 + row) * 513 + col),
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    c0_a2 = _a2_value(a2_ptr, a, 0, row + 1, col - 257, valid & (c == 0) & (row < 255) & (col >= 258))
    full8 = tl.load(
        arg17_ptr + (a * 255 + row) * 513 + col,
        mask=valid & (c == 0) & (row < 255) & (col < 258),
        other=0.0,
    ).to(tl.float32)
    full9_c0 = tl.load(
        arg18_ptr + (a * 512 + row) * 513 + col,
        mask=valid & (c == 0) & (row >= 255),
        other=0.0,
    ).to(tl.float32)
    c0_edge = tl.where(row < 255, tl.where(col >= 258, c0_a2, full8), full9_c0)
    term0 = tl.where(c == 0, c0_edge, full10)

    d1_mid = (row >= 255) & (row < 511)
    d1_b1 = _b1_value(a2_ptr, arg16_ptr, a, c + 1, row - 255, col - 257, valid & d1_mid & (col >= 257))
    d1_full12 = tl.load(
        arg21_ptr + (((a * 3 + c) * 256 + (row - 255)) * 513 + col),
        mask=valid & d1_mid & (col < 257),
        other=0.0,
    ).to(tl.float32)
    term1 = tl.where(d1_mid, tl.where(col >= 257, d1_b1, d1_full12), full10)

    c2_row = row >= 256
    c2_b2 = _b2_value(a2_ptr, arg16_ptr, arg20_ptr, a, 3, row - 256, col + 256, valid & (c == 2) & c2_row & (col <= 256))
    c2_full14 = tl.load(
        arg23_ptr + (a * 256 + (row - 256)) * 513 + col,
        mask=valid & (c == 2) & c2_row & (col > 256),
        other=0.0,
    ).to(tl.float32)
    c2_full9 = tl.load(
        arg18_ptr + (a * 512 + row) * 513 + col,
        mask=valid & (c == 2) & (~c2_row),
        other=0.0,
    ).to(tl.float32)
    c2_edge = tl.where(c2_row, tl.where(col <= 256, c2_b2, c2_full14), c2_full9)
    term2 = tl.where(c == 2, c2_edge, full10)

    c3_row = row < 256
    c3_b3 = _b3_value(a2_ptr, arg16_ptr, arg20_ptr, arg22_ptr, a, c, row, col + 256, valid & c3_row & (col <= 256))
    c3_full12 = tl.load(
        arg21_ptr + (((a * 3 + c) * 256 + row) * 513 + col),
        mask=valid & c3_row & (col > 256),
        other=0.0,
    ).to(tl.float32)
    term3 = tl.where(c3_row, tl.where(col <= 256, c3_b3, c3_full12), full10)

    acc01 = _bf16_add_f32(term0, term1)
    acc012 = _bf16_add_f32(acc01, term2)
    value = _bf16_add_f32(acc012, term3)
    tl.store(out_ptr + offsets, value.to(tl.bfloat16), mask=valid)


@oracle_impl(hardware="B200", point="c623eb69", BLOCK_R_=BLOCK_R, BLOCK_=BLOCK_OUT, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R_: int, BLOCK_: int, num_warps: int):
    (
        arg0_1,
        _arg1_1,
        _arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        *_shape_params,
    ) = inputs

    product = torch.empty((8, 1024, 12, 513), device=arg0_1.device, dtype=torch.float32)
    probs = torch.empty((8, 1024, 12, 513), device=arg0_1.device, dtype=torch.float32)
    _product_probs_kernel[(TOTAL_A2_ROWS,)](
        arg0_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        product,
        probs,
        BLOCK_R_=BLOCK_R_,
        num_warps=num_warps,
        num_stages=3,
    )
    row_sums = torch.ops.aten.sum.dim_IntList(product, [-1], False)

    a2 = torch.empty((96, 4, 256, 513), device=arg0_1.device, dtype=torch.bfloat16)
    _a2_from_product_kernel[(TOTAL_A2_ROWS,)](
        product,
        probs,
        row_sums,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        a2,
        BLOCK_R_=BLOCK_R_,
        num_warps=num_warps,
        num_stages=3,
    )

    out = torch.empty((288, 512, 512), device=arg0_1.device, dtype=torch.bfloat16)
    _final_output_kernel[(triton.cdiv(TOTAL_OUT, BLOCK_),)](
        a2,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        out,
        TOTAL_=TOTAL_OUT,
        BLOCK_=BLOCK_,
        num_warps=4,
        num_stages=3,
    )
    return out
