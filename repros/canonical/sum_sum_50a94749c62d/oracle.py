"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 slice-add BatchNorm-backward tail by co-reducing the rounded low-channel residual producer into both returned per-channel sums, finalizing the scale-gradient vector and dependent channel coefficients once, and writing the channels-last bf16 input-gradient epilogue directly, whereas Inductor schedules the slice/add cast, sibling channel reductions, vector epilogue, and full-tensor epilogue as generic reduction and pointwise regions over materialized intermediates; Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template that carries compatible summaries into a dependent dense epilogue while preserving bf16 add and final cast boundaries; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward channel-reduction lowering that shares split-K partials for both sums and sinks the returned vector and dense epilogues into one coordinated plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 512
IN_CHANNELS = 112
CHANNELS = 56
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
NUMEL = ELEMENTS_PER_CHANNEL * CHANNELS
REDUCE_SCALE = 9.964923469387754e-06
OUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (CHANNELS * HW, 1, CHANNELS * WIDTH, CHANNELS)
_USE_INDUCTOR_NUMERICS = False


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
def _bf16_add_to_f32(a, b):
    return _f32_add(a, b).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _partial_reduce_kernel(
    wide_ptr,
    residual_ptr,
    x_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_prod_ptr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    active = (rows[:, None] < 100352) & (cols[None, :] < 56)

    wide_offsets = rows[:, None] * 112 + cols[None, :]
    narrow_offsets = rows[:, None] * 56 + cols[None, :]

    lhs = tl.load(wide_ptr + wide_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(residual_ptr + narrow_offsets, mask=active, other=0.0).to(tl.float32)
    if USE_INDUCTOR_NUMERICS:
        add_value = _f32_add(lhs, rhs)
    else:
        add_value = _bf16_add_to_f32(lhs, rhs)

    x = tl.load(x_ptr + narrow_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + cols, mask=cols < 56, other=0.0).to(tl.float32)
    centered = _f32_sub(x, mean[None, :])
    prod = _f32_mul(add_value, centered)

    partial_offsets = tl.program_id(1) * 56 + cols
    tl.store(
        partial_sum_ptr + partial_offsets,
        tl.sum(tl.where(active, add_value, 0.0), axis=0),
        mask=cols < 56,
    )
    tl.store(
        partial_prod_ptr + partial_offsets,
        tl.sum(tl.where(active, prod, 0.0), axis=0),
        mask=cols < 56,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_prod_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    prod_tmp_ptr,
    prod_scaled_out_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    active = (chunks[:, None] < NUM_CHUNKS) & (cols[None, :] < 56)
    offsets = chunks[:, None] * 56 + cols[None, :]

    partial_sum = tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    partial_prod = tl.load(partial_prod_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum_value = tl.sum(partial_sum, axis=0).to(tl.float32)
    prod_value = tl.sum(partial_prod, axis=0).to(tl.float32)

    col_mask = cols < 56
    invstd = tl.load(invstd_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    mean_term = _f32_mul(sum_value, 9.964923469387754e-06)
    prod_scaled = _f32_mul(prod_value, 9.964923469387754e-06)
    invstd_sq = _f32_mul(invstd, invstd)
    prod_coeff = _f32_mul(prod_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)
    prod_scaled_out = _f32_mul(prod_value, invstd)

    tl.store(sum_out_ptr + cols, sum_value, mask=col_mask)
    tl.store(prod_tmp_ptr + cols, prod_value, mask=col_mask)
    tl.store(prod_scaled_out_ptr + cols, prod_scaled_out, mask=col_mask)
    tl.store(mean_term_ptr + cols, mean_term, mask=col_mask)
    tl.store(prod_coeff_ptr + cols, prod_coeff, mask=col_mask)
    tl.store(output_scale_ptr + cols, output_scale, mask=col_mask)


@triton.jit
def _epilogue_kernel(
    wide_ptr,
    residual_ptr,
    x_ptr,
    mean_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < 5623808
    channel = offsets - (offsets // 56) * 56
    reduce_index = offsets // 56
    wide_offsets = reduce_index * 112 + channel

    lhs = tl.load(wide_ptr + wide_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if USE_INDUCTOR_NUMERICS:
        add_value = _f32_add(lhs, rhs)
    else:
        add_value = _bf16_add_to_f32(lhs, rhs)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    centered = _f32_sub(x, mean)

    prod_coeff = tl.load(prod_coeff_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    correction = _f32_mul(centered, prod_coeff)
    residual = _f32_sub(add_value, correction)

    mean_term = tl.load(mean_term_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    residual = _f32_sub(residual, mean_term)

    output_scale = tl.load(output_scale_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    out_value = _f32_mul(residual, output_scale)
    tl.store(out_ptr + offsets, out_value.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@oracle_impl(
    hardware="B200",
    point="fe80348b",
    BLOCK_R=1024,
    BLOCK_C=16,
    BLOCK_E=1024,
    num_warps_reduce=8,
    num_warps_final=8,
    num_warps_epilogue=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_E: int,
    num_warps_reduce: int,
    num_warps_final: int,
    num_warps_epilogue: int,
):
    global _USE_INDUCTOR_NUMERICS
    arg0, arg1, arg2, mean, invstd, weight = inputs
    device = arg0.device
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True
    num_chunks = triton.cdiv(ELEMENTS_PER_CHANNEL, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    partial_sum = torch.empty_strided((num_chunks, CHANNELS), (CHANNELS, 1), device=device, dtype=torch.float32)
    partial_prod = torch.empty_strided((num_chunks, CHANNELS), (CHANNELS, 1), device=device, dtype=torch.float32)
    sum_out = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    prod_tmp = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    prod_scaled_out = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    mean_term = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    output_scale = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=device, dtype=torch.bfloat16)

    _partial_reduce_kernel[(triton.cdiv(CHANNELS, BLOCK_C), num_chunks)](
        arg0,
        arg1,
        arg2,
        mean,
        partial_sum,
        partial_prod,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps_reduce,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(CHANNELS, BLOCK_C),)](
        partial_sum,
        partial_prod,
        invstd,
        weight,
        sum_out,
        prod_tmp,
        prod_scaled_out,
        mean_term,
        prod_coeff,
        output_scale,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_final,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, BLOCK_E),)](
        arg0,
        arg1,
        arg2,
        mean,
        mean_term,
        prod_coeff,
        output_scale,
        out,
        BLOCK=BLOCK_E,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps_epilogue,
        num_stages=4,
    )
    return sum_out, prod_scaled_out, out
