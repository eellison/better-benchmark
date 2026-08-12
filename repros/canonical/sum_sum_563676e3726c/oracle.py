"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ShuffleNet bf16 channel-shuffle plus masked batchnorm-backward scope by materializing the returned shuffled add tensor, cooperatively reducing the high-half slice and centered activation over the shared N/H/W domain, and reusing the finalized summaries for the f32 vector output and channels-last bf16 epilogue, whereas Inductor schedules the add/view/permute/clone/view producer, sibling channel reductions, and dependent tensor epilogue as separate generic regions around materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no full-scope cooperative split-K template that keeps a static channel-shuffle producer and compatible BN-backward reductions coordinated with their dependent epilogue; the fix is COOPERATIVE_SPLIT_K: teach Inductor to inline static ShuffleNet channel-shuffle producers into split channel reductions, finalize shared summaries once, and sink the vector/tensor epilogues while preserving bf16 cast boundaries and output strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 116
C_OUT = 232
H = 14
W = 14
HW = H * W
R = N * HW
NUMEL_DENSE = N * C * HW
INV_R = 3.985969387755102e-05


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
def _shuffled_add_value(
    arg0_ptr,
    arg1_ptr,
    n,
    spatial,
    c,
    branch,
    active,
    C_OUT_: tl.constexpr,
    HW_: tl.constexpr,
):
    in_c = c * 2 + branch
    offset = n * C_OUT_ * HW_ + spatial * C_OUT_ + in_c
    lhs = tl.load(arg0_ptr + offset, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offset, mask=active, other=0.0).to(tl.float32)
    return _f32_add(lhs, rhs).to(tl.bfloat16, fp_downcast_rounding="rtne")


@triton.jit
def _selected_and_centered_from_source(
    source,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    n,
    spatial,
    c,
    active,
    C_: tl.constexpr,
    HW_: tl.constexpr,
):
    dense_offset = n * C_ * HW_ + spatial * C_ + c
    activation = tl.load(activation_ptr + dense_offset, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean)
    affine = _f32_add(_f32_mul(_f32_mul(centered, invstd), weight), bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    fill = tl.load(fill_ptr)
    selected = tl.where(affine_bf16 <= 0.0, fill, source).to(tl.float32)
    return selected, centered


@triton.jit
def _partial_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    shuffled_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    R_: tl.constexpr,
    C_: tl.constexpr,
    C_OUT_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    c_block = tl.program_id(1)
    rows = tile * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_active = rows < R_
    col_active = cols < C_
    active = row_active[:, None] & col_active[None, :]
    n = rows // HW_
    spatial = rows - n * HW_

    low = _shuffled_add_value(
        arg0_ptr,
        arg1_ptr,
        n[:, None],
        spatial[:, None],
        cols[None, :],
        0,
        active,
        C_OUT_,
        HW_,
    )
    high = _shuffled_add_value(
        arg0_ptr,
        arg1_ptr,
        n[:, None],
        spatial[:, None],
        cols[None, :],
        1,
        active,
        C_OUT_,
        HW_,
    )
    out_base = n[:, None] * C_OUT_ * HW_ + cols[None, :] * HW_ + spatial[:, None]
    tl.store(shuffled_ptr + out_base, low, mask=active)
    tl.store(shuffled_ptr + out_base + C_ * HW_, high, mask=active)

    selected, centered = _selected_and_centered_from_source(
        high,
        activation_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        fill_ptr,
        n[:, None],
        spatial[:, None],
        cols[None, :],
        active,
        C_,
        HW_,
    )
    selected = tl.where(active, selected, 0.0)
    centered = tl.where(active, centered, 0.0)
    offsets = tile * C_ + cols
    tl.store(partial_sum_ptr + offsets, tl.sum(selected, axis=0), mask=col_active)
    tl.store(partial_dot_ptr + offsets, tl.sum(_f32_mul(selected, centered), axis=0), mask=col_active)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vec_out_ptr,
    mean_term_ptr,
    coeff_ptr,
    output_scale_ptr,
    C_: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    INV_R_: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < NUM_TILES
    offsets = tiles * C_ + c
    sum_value = tl.sum(tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32), axis=0)
    dot_value = tl.sum(tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32), axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    dot_scaled = _f32_mul(dot_value, INV_R_)
    invstd_sq = _f32_mul(invstd, invstd)
    tl.store(sum_out_ptr + c, sum_value)
    tl.store(vec_out_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, INV_R_))
    tl.store(coeff_ptr + c, _f32_mul(dot_scaled, invstd_sq))
    tl.store(output_scale_ptr + c, _f32_mul(invstd, weight))


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    mean_term_ptr,
    coeff_ptr,
    output_scale_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    C_OUT_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < NUMEL_
    c = linear % C_
    row = linear // C_
    n = row // HW_
    spatial = row - n * HW_

    high = _shuffled_add_value(
        arg0_ptr,
        arg1_ptr,
        n,
        spatial,
        c,
        1,
        active,
        C_OUT_,
        HW_,
    )
    selected, centered = _selected_and_centered_from_source(
        high,
        activation_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        fill_ptr,
        n,
        spatial,
        c,
        active,
        C_,
        HW_,
    )
    mean_term = tl.load(mean_term_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff = tl.load(coeff_ptr + c, mask=active, other=0.0).to(tl.float32)
    output_scale = tl.load(output_scale_ptr + c, mask=active, other=0.0).to(tl.float32)
    adjusted = _f32_sub(_f32_sub(selected, _f32_mul(centered, coeff)), mean_term)
    out = _f32_mul(adjusted, output_scale)
    tl.store(out_ptr + linear, out, mask=active)


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="f0fa1db9",
    BLOCK_R=128,
    BLOCK_C=32,
    EPILOGUE_BLOCK=512,
    reduce_warps=8,
    final_warps=4,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R,
    BLOCK_C,
    EPILOGUE_BLOCK,
    reduce_warps,
    final_warps,
    epilogue_warps,
):
    arg0, arg1, activation, mean, invstd, weight, bias, fill, _shape_param_0, _shape_param_1 = inputs
    device = arg0.device
    shuffled = torch.empty_strided(
        (N, C_OUT, H, W),
        (C_OUT * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    vec_out = torch.empty((C,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    num_tiles = triton.cdiv(R, BLOCK_R)
    partial_sum = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    coeff = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)

    _partial_reduce_kernel[(num_tiles, triton.cdiv(C, BLOCK_C))](
        arg0,
        arg1,
        activation,
        mean,
        invstd,
        weight,
        bias,
        fill,
        shuffled,
        partial_sum,
        partial_dot,
        R_=R,
        C_=C,
        C_OUT_=C_OUT,
        HW_=HW,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(C,)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        sum_out,
        vec_out,
        mean_term,
        coeff,
        output_scale,
        C_=C,
        NUM_TILES=num_tiles,
        BLOCK_TILES=_ceil_pow2(num_tiles),
        INV_R_=INV_R,
        num_warps=final_warps,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL_DENSE, EPILOGUE_BLOCK),)](
        arg0,
        arg1,
        activation,
        mean,
        invstd,
        weight,
        bias,
        fill,
        mean_term,
        coeff,
        output_scale,
        dense_out,
        NUMEL_=NUMEL_DENSE,
        C_=C,
        C_OUT_=C_OUT,
        HW_=HW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return shuffled, sum_out, vec_out, dense_out
