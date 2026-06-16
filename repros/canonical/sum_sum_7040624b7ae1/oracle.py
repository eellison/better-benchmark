"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 masked BN-backward-style branch, using the bf16-rounded sliced-add/where producer for both channel reductions and the returned dense epilogue, whereas Inductor schedules the masked producer, sibling reductions, coefficient math, and dense epilogue through generic reduction/pointwise regions; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output lowering that preserves the captured output contract while sharing finalized channel summaries with the dependent full-tensor epilogue; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward reduction lowering that splits the `N,H,W` dimension, finalizes compatible channel partials once, and sinks the dependent channels-last bf16 epilogue into the same planned lowering."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C_IN = 184
C = 92
H = 14
W = 14
HW = H * W
K_TOTAL = N * HW
TOTAL = N * C * HW
SCALE = 9.964923469387754e-06


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _masked_source_rounded(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    fill_ptr,
    n,
    hw,
    c,
    active,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
):
    offset0 = n * (C_IN_ * HW_) + hw * C_IN_ + c
    offset = n * (C_ * HW_) + hw * C_ + c
    lhs = tl.load(arg0_ptr + offset0, mask=active, other=0.0)
    rhs = tl.load(arg1_ptr + offset, mask=active, other=0.0)
    mask_value = tl.load(arg2_ptr + offset, mask=active, other=1.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    added = _f32_add(lhs.to(tl.float32), rhs.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    return tl.where(mask_value <= 0.0, fill, added).to(tl.float32)


@triton.jit
def _partial_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    fill_ptr,
    arg4_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    NUM_TILES: tl.constexpr,
    X_TOTAL: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    x_idx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r = tl.arange(0, BLOCK_K)[None, :]
    x_active = x_idx < X_TOTAL
    c = x_idx % C_
    tile = x_idx // C_
    k = tile * BLOCK_K + r
    active = x_active & (k < K_TOTAL_)
    n = k // HW_
    hw = k - n * HW_
    offset = n * (C_ * HW_) + hw * C_ + c

    source = _masked_source_rounded(
        arg0_ptr, arg1_ptr, arg2_ptr, fill_ptr, n, hw, c, active, C_IN_, C_, HW_
    )
    arg4_value = tl.load(arg4_ptr + offset, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _f32_sub(arg4_value, mean)

    source = tl.where(active, source, 0.0)
    centered = tl.where(active, centered, 0.0)
    partial_offset = x_idx
    tl.store(partial_sum_ptr + partial_offset, tl.sum(source, axis=1)[:, None], mask=x_active)
    tl.store(
        partial_dot_ptr + partial_offset,
        tl.sum(_f32_mul(source, centered), axis=1)[:, None],
        mask=x_active,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    mean_term_ptr,
    coeff_ptr,
    output_scale_ptr,
    SCALE_: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_TILES
    offsets = c + 92 * tiles
    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    invstd_sq = _f32_mul(invstd, invstd)
    scaled_dot = _f32_mul(dot_value, invstd)
    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scaled_dot_out_ptr + c, scaled_dot)
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, SCALE_))
    tl.store(coeff_ptr + c, _f32_mul(_f32_mul(dot_value, SCALE_), invstd_sq))
    tl.store(output_scale_ptr + c, _f32_mul(invstd, weight))


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    fill_ptr,
    arg4_ptr,
    mean_ptr,
    mean_term_ptr,
    coeff_ptr,
    output_scale_ptr,
    out_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < TOTAL_
    c = linear % C_
    k = linear // C_
    n = k // HW_
    hw = k - n * HW_

    source = _masked_source_rounded(
        arg0_ptr, arg1_ptr, arg2_ptr, fill_ptr, n, hw, c, active, C_IN_, C_, HW_
    )
    x = tl.load(arg4_ptr + linear, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered = _f32_sub(x, mean)
    coeff = tl.load(coeff_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = tl.load(mean_term_ptr + c, mask=active, other=0.0).to(tl.float32)
    output_scale = tl.load(output_scale_ptr + c, mask=active, other=0.0).to(tl.float32)
    correction = _f32_mul(centered, coeff)
    corrected = _f32_sub(source, correction)
    corrected = _f32_sub(corrected, mean_term)
    out = _f32_mul(corrected, output_scale)
    tl.store(out_ptr + linear, out, mask=active)


# 26501e14: (T([512,184,14,14], bf16, stride=(36064,1,2576,184)), T([512,92,14,14], bf16, stride=(18032,1,1288,92)) x3, T([], bf16), T([512,92,14,14], bf16), T([1,92,1,1], f32), T([92], f32), T([92], f32))
@oracle_impl(
    hardware="B200",
    point="26501e14",
    BLOCK_K=128,
    XBLOCK=16,
    EPILOGUE_BLOCK=256,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    XBLOCK: int,
    EPILOGUE_BLOCK: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    device = arg0_1.device
    num_tiles = triton.cdiv(K_TOTAL, BLOCK_K)
    block_tiles = _next_power_of_2(num_tiles)
    x_total = C * num_tiles

    partial_sum = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    coeff = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(x_total, XBLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        partial_sum,
        partial_dot,
        C_IN_=C_IN,
        C_=C,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        NUM_TILES=num_tiles,
        X_TOTAL=x_total,
        XBLOCK=XBLOCK,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=4,
    )
    _finalize_kernel[(C,)](
        partial_sum,
        partial_dot,
        arg6_1,
        arg7_1,
        sum_out,
        scaled_dot_out,
        mean_term,
        coeff,
        output_scale,
        SCALE_=SCALE,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
        num_stages=1,
    )
    _epilogue_kernel[(triton.cdiv(TOTAL, EPILOGUE_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        mean_term,
        coeff,
        output_scale,
        out,
        C_IN_=C_IN,
        C_=C,
        HW_=HW,
        TOTAL_=TOTAL,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return sum_out, scaled_dot_out, out
