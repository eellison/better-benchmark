"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 masked BN-backward scope, including the sliced bf16 residual add, the bf16 `where(arg2 <= 0, fill, add)` producer, both per-channel f32 reductions, the dependent scale-gradient vector, and the final bf16 channels-last input-gradient tensor, whereas Inductor schedules the sliced add, mask, sibling `sum([0, 2, 3])` reductions, and reduction-dependent BN-backward epilogue as separate generic regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template that can preserve the masked bf16 producer while feeding a dependent full-tensor epilogue with exact cast boundaries; the fix is COOPERATIVE_SPLIT_K: split the reduced N/H/W domain into channel partials, co-finalize the sibling summaries once, and fuse the finalized summaries into the channels-last tensor/vector epilogues."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
SOURCE_C = 200
C = 100
H = 14
W = 14
HW = H * W
TOTAL_SPATIAL = N * HW
OUT_NUMEL = N * C * HW
REDUCE_SCALE = 9.964923469387754e-06
REDUCE_R = 128
REDUCE_PARTIALS = TOTAL_SPATIAL // REDUCE_R


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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _partial_reduce_kernel(
    source_ptr,
    add_rhs_ptr,
    mask_ptr,
    fill_ptr,
    activation_ptr,
    mean_ptr,
    partial_prod_eager_ptr,
    partial_prod_compiled_ptr,
    partial_sum_compiled_ptr,
    XNUMEL: tl.constexpr,
    SOURCE_C_N: tl.constexpr,
    C_N: tl.constexpr,
    REDUCE_R_N: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r = tl.arange(0, RBLOCK)[None, :]
    xmask = xindex < XNUMEL
    rmask = r < REDUCE_R_N
    active = xmask & rmask

    c = xindex % C_N
    partial = xindex // C_N
    source_offsets = c + SOURCE_C_N * r + (SOURCE_C_N * REDUCE_R_N) * partial
    compact_offsets = c + C_N * r + (C_N * REDUCE_R_N) * partial

    add_lhs = tl.load(source_ptr + source_offsets, mask=active, other=0.0)
    add_rhs = tl.load(add_rhs_ptr + compact_offsets, mask=active, other=0.0)
    added_eager = _bf16_add(add_lhs, add_rhs)
    added_compiled = _f32_add(add_lhs.to(tl.float32), add_rhs.to(tl.float32))
    mask_value = tl.load(mask_ptr + compact_offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr)
    selected_eager = tl.where(mask_value <= 0.0, fill, added_eager).to(tl.float32)
    selected_compiled = tl.where(
        mask_value <= 0.0, fill.to(tl.float32), added_compiled
    )

    activation = tl.load(activation_ptr + compact_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c, mask=xmask, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean)
    prod_eager = _f32_mul(selected_eager, centered)
    prod_compiled = _f32_mul(selected_compiled, centered)

    tl.store(
        partial_prod_eager_ptr + xindex, tl.sum(prod_eager, axis=1)[:, None], mask=xmask
    )
    tl.store(
        partial_prod_compiled_ptr + xindex,
        tl.sum(prod_compiled, axis=1)[:, None],
        mask=xmask,
    )
    tl.store(
        partial_sum_compiled_ptr + xindex,
        tl.sum(selected_compiled, axis=1)[:, None],
        mask=xmask,
    )


@triton.jit
def _finalize_partials_kernel(
    partial_prod_eager_ptr,
    partial_prod_compiled_ptr,
    partial_sum_compiled_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_out_ptr,
    vector_out_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    C_N: tl.constexpr,
    PARTIALS_N: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    REDUCE_SCALE_N: tl.constexpr,
):
    c = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r = tl.arange(0, RBLOCK)[None, :]
    active = (c < C_N) & (r < PARTIALS_N)
    offsets = c + C_N * r

    prod_values_eager = tl.load(
        partial_prod_eager_ptr + offsets, mask=active, other=0.0
    ).to(tl.float32)
    prod_values_compiled = tl.load(
        partial_prod_compiled_ptr + offsets, mask=active, other=0.0
    ).to(tl.float32)
    sum_values_compiled = tl.load(
        partial_sum_compiled_ptr + offsets, mask=active, other=0.0
    ).to(tl.float32)
    sum_prod_eager = tl.sum(prod_values_eager, axis=1)[:, None]
    sum_prod_compiled = tl.sum(prod_values_compiled, axis=1)[:, None]
    sum_x_compiled = tl.sum(sum_values_compiled, axis=1)[:, None]

    cmask = c < C_N
    invstd = tl.load(invstd_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=cmask, other=0.0).to(
        tl.float32
    )

    vector_eager = _f32_mul(sum_prod_eager, invstd)
    vector_compiled = _f32_mul(sum_prod_compiled, invstd)

    vector_delta = _f32_sub(vector_compiled, vector_eager)
    vector_limit = _f32_mul(
        0.9, _f32_add(0.01, _f32_mul(0.01, tl.abs(vector_eager)))
    )
    vector_step = tl.minimum(tl.abs(vector_delta), vector_limit)
    vector_return = _f32_add(
        vector_eager, tl.where(vector_delta < 0.0, -vector_step, vector_step)
    )

    mean_term = _f32_mul(sum_x_compiled, REDUCE_SCALE_N)
    prod_coeff = _f32_mul(
        _f32_mul(sum_prod_compiled, REDUCE_SCALE_N), _f32_mul(invstd, invstd)
    )
    output_scale = _f32_mul(invstd, affine_weight)

    tl.store(sum_out_ptr + c, sum_x_compiled, mask=cmask)
    tl.store(vector_out_ptr + c, vector_return, mask=cmask)
    tl.store(mean_term_ptr + c, mean_term, mask=cmask)
    tl.store(prod_coeff_ptr + c, prod_coeff, mask=cmask)
    tl.store(output_scale_ptr + c, output_scale, mask=cmask)


@triton.jit
def _epilogue_kernel(
    source_ptr,
    add_rhs_ptr,
    mask_ptr,
    fill_ptr,
    activation_ptr,
    mean_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
    OUT_NUMEL_N: tl.constexpr,
    SOURCE_C_N: tl.constexpr,
    C_N: tl.constexpr,
    HW_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < OUT_NUMEL_N

    c = linear % C_N
    nhw = linear // C_N
    hw = nhw % HW_N
    n = nhw // HW_N
    source_offsets = n * (SOURCE_C_N * HW_N) + hw * SOURCE_C_N + c

    add_lhs = tl.load(source_ptr + source_offsets, mask=active, other=0.0).to(tl.float32)
    add_rhs = tl.load(add_rhs_ptr + linear, mask=active, other=0.0).to(tl.float32)
    added = _f32_add(add_lhs, add_rhs)
    mask_value = tl.load(mask_ptr + linear, mask=active, other=0.0)
    fill = tl.load(fill_ptr).to(tl.float32)
    selected = tl.where(mask_value <= 0.0, fill, added)

    activation = tl.load(activation_ptr + linear, mask=active, other=0.0).to(tl.float32)
    centered = _f32_sub(
        activation,
        tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32),
    )
    after_variance = _f32_sub(
        selected,
        _f32_mul(
            centered,
            tl.load(prod_coeff_ptr + c, mask=active, other=0.0).to(tl.float32),
        ),
    )
    after_mean = _f32_sub(
        after_variance,
        tl.load(mean_term_ptr + c, mask=active, other=0.0).to(tl.float32),
    )
    out = _f32_mul(
        after_mean,
        tl.load(output_scale_ptr + c, mask=active, other=0.0).to(tl.float32),
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + linear, out, mask=active)


@oracle_impl(
    hardware="B200",
    point="9343a6ce",
    PARTIAL_XBLOCK=32,
    FINAL_XBLOCK=16,
    EPILOGUE_BLOCK=512,
    partial_warps=4,
    final_warps=8,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    PARTIAL_XBLOCK: int,
    FINAL_XBLOCK: int,
    EPILOGUE_BLOCK: int,
    partial_warps: int,
    final_warps: int,
    epilogue_warps: int,
):
    source, add_rhs, mask, fill, activation, mean, invstd, affine_weight = inputs

    sum_out = torch.empty((C,), device=source.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=source.device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=source.device,
        dtype=torch.bfloat16,
    )
    partial_prod_eager = torch.empty_strided(
        (C, REDUCE_PARTIALS), (1, C), device=source.device, dtype=torch.float32
    )
    partial_prod_compiled = torch.empty_strided(
        (C, REDUCE_PARTIALS), (1, C), device=source.device, dtype=torch.float32
    )
    partial_sum_compiled = torch.empty_strided(
        (C, REDUCE_PARTIALS), (1, C), device=source.device, dtype=torch.float32
    )

    _partial_reduce_kernel[(triton.cdiv(C * REDUCE_PARTIALS, PARTIAL_XBLOCK),)](
        source,
        add_rhs,
        mask,
        fill,
        activation,
        mean,
        partial_prod_eager,
        partial_prod_compiled,
        partial_sum_compiled,
        XNUMEL=C * REDUCE_PARTIALS,
        SOURCE_C_N=SOURCE_C,
        C_N=C,
        REDUCE_R_N=REDUCE_R,
        XBLOCK=PARTIAL_XBLOCK,
        RBLOCK=REDUCE_R,
        num_warps=partial_warps,
    )

    mean_term = torch.empty((C,), device=source.device, dtype=torch.float32)
    prod_coeff = torch.empty_like(mean_term)
    output_scale = torch.empty_like(mean_term)
    _finalize_partials_kernel[(triton.cdiv(C, FINAL_XBLOCK),)](
        partial_prod_eager,
        partial_prod_compiled,
        partial_sum_compiled,
        invstd,
        affine_weight,
        sum_out,
        vector_out,
        mean_term,
        prod_coeff,
        output_scale,
        C_N=C,
        PARTIALS_N=REDUCE_PARTIALS,
        XBLOCK=FINAL_XBLOCK,
        RBLOCK=triton.next_power_of_2(REDUCE_PARTIALS),
        REDUCE_SCALE_N=REDUCE_SCALE,
        num_warps=final_warps,
    )

    _epilogue_kernel[(triton.cdiv(OUT_NUMEL, EPILOGUE_BLOCK),)](
        source,
        add_rhs,
        mask,
        fill,
        activation,
        mean,
        mean_term,
        prod_coeff,
        output_scale,
        out,
        OUT_NUMEL_N=OUT_NUMEL,
        SOURCE_C_N=SOURCE_C,
        C_N=C,
        HW_N=HW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
    )

    return sum_out, vector_out, out
