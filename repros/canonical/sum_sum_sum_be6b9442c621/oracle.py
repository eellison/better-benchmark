"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete PyTorch U-Net bf16 bilinear-upsample-backward scatter feeding BN/ReLU masking and three channel reductions, preserving the real row/column index tensors, duplicate-preserving `index_put(accumulate=True)` behavior, the bf16 scatter/mask/BN-backward cast boundaries, the returned dense `[1,128,160,239]` tensor, and the final bf16-rounded channel sum; Inductor currently lowers the four generic scatter-adds, bf16 masking producer, dependent reductions, and final bf16 output reduction as separate materialized kernels; Inductor cannot do this today because scheduler/codegen does not recognize indexed bilinear-upsample-backward scatter-add producers as channel-reduction inputs with a required full-tensor epilogue; the fix is SCATTER_REDUCE: add a guarded lowering for duplicate-safe indexed bilinear scatter-reduces that fuses the downstream channel reductions plus returned bf16 store."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


C = 128
SRC_H = 320
SRC_W = 478
SRC_STRIDE_H = 479
OUT_H = 160
OUT_W = 239
OUT_HW = OUT_H * OUT_W
SCALE = 2.615062761506276e-05


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
def _to_bf16_f32(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _zero4_kernel(
    scatter0_ptr,
    scatter1_ptr,
    scatter2_ptr,
    scatter3_ptr,
    N_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N_
    zeros = tl.zeros((BLOCK_N,), dtype=tl.float32)
    tl.store(scatter0_ptr + offs, zeros, mask=mask)
    tl.store(scatter1_ptr + offs, zeros, mask=mask)
    tl.store(scatter2_ptr + offs, zeros, mask=mask)
    tl.store(scatter3_ptr + offs, zeros, mask=mask)


@triton.jit
def _bilinear_scatter_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    scatter0_ptr,
    scatter1_ptr,
    scatter2_ptr,
    scatter3_ptr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < 152960
    h = offs // 478
    w = offs - h * 478

    src_offsets = (c + 128) * 153280 + h * 479 + w
    x = tl.load(arg0_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    scale_h = tl.load(arg1_ptr + h, mask=mask, other=0.0).to(tl.float32)
    scale_w = tl.load(arg2_ptr + w, mask=mask, other=0.0).to(tl.float32)

    mul = _f32_mul(x, scale_h)
    add = _f32_add(x, -mul)
    mul_1 = _f32_mul(mul, scale_w)
    add_1 = _f32_add(mul, -mul_1)
    mul_2 = _f32_mul(add, scale_w)
    add_2 = _f32_add(add, -mul_2)

    row0 = tl.load(arg3_ptr + h, mask=mask, other=0).to(tl.int64)
    col0 = tl.load(arg4_ptr + w, mask=mask, other=0).to(tl.int64)
    col1 = tl.load(arg5_ptr + w, mask=mask, other=0).to(tl.int64)
    row1 = tl.load(arg6_ptr + h, mask=mask, other=0).to(tl.int64)

    base = c * 38240
    dst00 = base + row0 * 239 + col0
    dst01 = base + row0 * 239 + col1
    dst10 = base + row1 * 239 + col0
    dst11 = base + row1 * 239 + col1
    tl.atomic_add(scatter0_ptr + dst00, mul_1, sem="relaxed", mask=mask)
    tl.atomic_add(scatter1_ptr + dst01, add_1, sem="relaxed", mask=mask)
    tl.atomic_add(scatter2_ptr + dst10, mul_2, sem="relaxed", mask=mask)
    tl.atomic_add(scatter3_ptr + dst11, add_2, sem="relaxed", mask=mask)


@triton.jit
def _combine_scatter_kernel(
    scatter0_ptr,
    scatter1_ptr,
    scatter2_ptr,
    scatter3_ptr,
    scatter_bf16_ptr,
    N_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N_
    v0 = tl.load(scatter0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(scatter1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v2 = tl.load(scatter2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v3 = tl.load(scatter3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    add_3 = _f32_add(v0, v1)
    add_4 = _f32_add(add_3, v2)
    add_5 = _f32_add(add_4, v3)
    tl.store(scatter_bf16_ptr + offs, add_5.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@triton.jit
def _sum12_partial_kernel(
    arg7_ptr,
    arg8_ptr,
    arg9_ptr,
    arg10_ptr,
    arg11_ptr,
    arg12_ptr,
    scatter_bf16_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
    TILES_: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    hw = tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw < 38240
    offsets = c * 38240 + hw

    x7 = tl.load(arg7_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(arg8_ptr + c).to(tl.float32)
    invstd = tl.load(arg9_ptr + c).to(tl.float32)
    weight = tl.load(arg10_ptr + c).to(tl.float32)
    bias = tl.load(arg11_ptr + c).to(tl.float32)
    scalar = tl.load(arg12_ptr).to(tl.float32)
    scatter = tl.load(scatter_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x7, mean)
    gate_mul = _f32_mul(_f32_mul(centered, invstd), weight)
    gate = _to_bf16_f32(_f32_add(gate_mul, bias))
    active = gate <= 0.0
    y = tl.where(active, scalar, scatter)

    sum1 = tl.sum(tl.where(mask, y, 0.0), axis=0)
    sum2 = tl.sum(tl.where(mask, _f32_mul(y, centered), 0.0), axis=0)
    out_offset = c * TILES_ + tile
    tl.store(partial_sum1_ptr + out_offset, sum1)
    tl.store(partial_sum2_ptr + out_offset, sum2)


@triton.jit
def _sum12_finalize_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    out_sum1_ptr,
    sum2_ptr,
    out_mul13_ptr,
    arg9_ptr,
    TILES_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < TILES_
    base = c * TILES_ + offs
    vals1 = tl.load(partial_sum1_ptr + base, mask=mask, other=0.0).to(tl.float32)
    vals2 = tl.load(partial_sum2_ptr + base, mask=mask, other=0.0).to(tl.float32)
    sum1 = tl.sum(vals1, axis=0)
    sum2 = tl.sum(vals2, axis=0)
    invstd = tl.load(arg9_ptr + c).to(tl.float32)
    tl.store(out_sum1_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(out_mul13_ptr + c, _f32_mul(sum2, invstd))


@triton.jit
def _final_output_partial_kernel(
    arg7_ptr,
    arg8_ptr,
    arg9_ptr,
    arg10_ptr,
    arg11_ptr,
    arg12_ptr,
    scatter_bf16_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    partial_sum4_ptr,
    TILES_: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    hw = tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw < 38240
    offsets = c * 38240 + hw

    x7 = tl.load(arg7_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(arg8_ptr + c).to(tl.float32)
    invstd = tl.load(arg9_ptr + c).to(tl.float32)
    weight = tl.load(arg10_ptr + c).to(tl.float32)
    bias = tl.load(arg11_ptr + c).to(tl.float32)
    scalar = tl.load(arg12_ptr).to(tl.float32)
    scatter = tl.load(scatter_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c).to(tl.float32)

    centered = _f32_sub(x7, mean)
    gate_mul = _f32_mul(_f32_mul(centered, invstd), weight)
    gate = _to_bf16_f32(_f32_add(gate_mul, bias))
    active = gate <= 0.0
    y = tl.where(active, scalar, scatter)

    mean_term = _f32_mul(sum1, 2.615062761506276e-05)
    scaled_sum2 = _f32_mul(sum2, 2.615062761506276e-05)
    invstd_sq = _f32_mul(invstd, invstd)
    var_term = _f32_mul(scaled_sum2, invstd_sq)
    affine_scale = _f32_mul(invstd, weight)

    mul11 = _f32_mul(centered, var_term)
    sub2 = _f32_sub(y, mul11)
    sub3 = _f32_sub(sub2, mean_term)
    out_f32 = _f32_mul(sub3, affine_scale)
    out_bf16 = out_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + offsets, out_bf16, mask=mask)

    out_round = out_bf16.to(tl.float32)
    sum4 = tl.sum(tl.where(mask, out_round, 0.0), axis=0)
    tl.store(partial_sum4_ptr + c * TILES_ + tile, sum4)


@triton.jit
def _sum4_finalize_kernel(
    partial_sum4_ptr,
    out_sum4_ptr,
    TILES_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < TILES_
    vals = tl.load(partial_sum4_ptr + c * TILES_ + offs, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_sum4_ptr + c, rounded)


# (T([1,256,320,479], bf16), T([320,1], f32), T([478], f32), T([320,1], i64), T([478], i64), T([478], i64), T([320,1], i64), T([1,128,160,239], bf16), T([1,128,1,1], f32), T([1,128,1,1], f32), T([128], f32), T([128], f32), T([], bf16), S([1,128,160,239]))
@oracle_impl(
    hardware="B200",
    point="0ba1fdea",
    SRC_BLOCK=256,
    HW_BLOCK=512,
    ZERO_BLOCK=1024,
    scatter_warps=8,
    num_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    SRC_BLOCK: int,
    HW_BLOCK: int,
    ZERO_BLOCK: int,
    scatter_warps: int,
    num_warps: int,
    final_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
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
        _shape_param_0,
    ) = inputs
    del _shape_param_0

    device = arg0_1.device
    src_tiles = triton.cdiv(SRC_H * SRC_W, SRC_BLOCK)
    hw_tiles = triton.cdiv(OUT_HW, HW_BLOCK)

    scatter0 = torch.empty((C, OUT_HW), device=device, dtype=torch.float32)
    scatter1 = torch.empty((C, OUT_HW), device=device, dtype=torch.float32)
    scatter2 = torch.empty((C, OUT_HW), device=device, dtype=torch.float32)
    scatter3 = torch.empty((C, OUT_HW), device=device, dtype=torch.float32)
    scatter_bf16 = torch.empty((C, OUT_HW), device=device, dtype=torch.bfloat16)
    _zero4_kernel[(triton.cdiv(C * OUT_HW, ZERO_BLOCK),)](
        scatter0,
        scatter1,
        scatter2,
        scatter3,
        N_=C * OUT_HW,
        BLOCK_N=ZERO_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _bilinear_scatter_kernel[(C, src_tiles)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        scatter0,
        scatter1,
        scatter2,
        scatter3,
        BLOCK_K=SRC_BLOCK,
        num_warps=scatter_warps,
        num_stages=3,
    )
    _combine_scatter_kernel[(triton.cdiv(C * OUT_HW, ZERO_BLOCK),)](
        scatter0,
        scatter1,
        scatter2,
        scatter3,
        scatter_bf16,
        N_=C * OUT_HW,
        BLOCK_N=ZERO_BLOCK,
        num_warps=4,
        num_stages=3,
    )

    partial_sum1 = torch.empty((C, hw_tiles), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((C, hw_tiles), device=device, dtype=torch.float32)
    _sum12_partial_kernel[(C, hw_tiles)](
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        scatter_bf16,
        partial_sum1,
        partial_sum2,
        TILES_=hw_tiles,
        BLOCK_HW=HW_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )

    out_sum1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=device, dtype=torch.float32)
    out_mul13 = torch.empty((C,), device=device, dtype=torch.float32)
    _sum12_finalize_kernel[(C,)](
        partial_sum1,
        partial_sum2,
        out_sum1,
        sum2,
        out_mul13,
        arg9_1,
        TILES_=hw_tiles,
        BLOCK_TILES=triton.next_power_of_2(hw_tiles),
        num_warps=final_warps,
        num_stages=3,
    )

    out_tensor = torch.empty_strided(
        (1, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial_sum4 = torch.empty((C, hw_tiles), device=device, dtype=torch.float32)
    _final_output_partial_kernel[(C, hw_tiles)](
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        scatter_bf16,
        out_sum1,
        sum2,
        out_tensor,
        partial_sum4,
        TILES_=hw_tiles,
        BLOCK_HW=HW_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )

    out_sum4 = torch.empty((C,), device=device, dtype=torch.float32)
    _sum4_finalize_kernel[(C,)](
        partial_sum4,
        out_sum4,
        TILES_=hw_tiles,
        BLOCK_TILES=triton.next_power_of_2(hw_tiles),
        num_warps=final_warps,
        num_stages=3,
    )

    return out_sum1, out_mul13, out_tensor, out_sum4
