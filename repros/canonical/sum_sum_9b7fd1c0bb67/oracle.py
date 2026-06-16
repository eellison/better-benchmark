"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete phlippe DenseNet bf16 BN-backward return scope by split-K reducing the shared masked `where(arg1 <= 0, arg2, arg3)` producer into the returned `sum(where)` and `sum(where * centered) * arg6` f32 channel vectors, then sinking the finalized summaries into the bf16 residual-add epilogue while returning the required `add[:, :16]` alias, whereas Inductor schedules the sibling reductions, broadcast BN-backward epilogue, bf16 residual add, and aliasing slice return as separate generic regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen has no cooperative multi-output channel-reduction plan that reuses the same masked/centered producer and finalized summaries while preserving bf16/f32 cast boundaries and output aliasing; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to split compatible BN-backward channel reductions over N/H/W, combine partial summaries once, and fuse the dependent dense/add epilogue plus alias-view return."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 7.62939453125e-06
SLICE_START = 16
EPILOGUE_BLOCK_C = 8


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
def _partial_reduce_kernel(
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    n_offsets = k_offsets // HW
    hw_offsets = k_offsets - n_offsets * HW
    offsets = n_offsets[:, None] * (C * HW) + c_offsets[None, :] * HW + hw_offsets[:, None]
    mask = (k_offsets[:, None] < K_TOTAL) & (c_offsets[None, :] < C)

    gate = tl.load(mask_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    selected = tl.where(gate <= 0.0, fill, rhs)

    activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean[None, :])
    dot = _f32_mul(selected, centered)

    partial_sum = tl.sum(tl.where(mask, selected, 0.0), axis=0)
    partial_dot = tl.sum(tl.where(mask, dot, 0.0), axis=0)
    out_offsets = tl.program_id(1) * C + c_offsets
    tl.store(partial_sum_ptr + out_offsets, partial_sum, mask=c_offsets < C)
    tl.store(partial_dot_ptr + out_offsets, partial_dot, mask=c_offsets < C)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    mean_term_ptr,
    correction_scale_ptr,
    output_scale_ptr,
    C: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
):
    c = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    mask = tile_offsets < NUM_K_TILES
    offsets = tile_offsets * C + c

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
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    correction_scale = _f32_mul(dot_mean, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scaled_dot_out_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, SCALE_VALUE))
    tl.store(correction_scale_ptr + c, correction_scale)
    tl.store(output_scale_ptr + c, output_scale)


@triton.jit
def _epilogue_kernel(
    residual_ptr,
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    mean_term_ptr,
    correction_scale_ptr,
    output_scale_ptr,
    out_ptr,
    C: tl.constexpr,
    INPUT_C: tl.constexpr,
    SLICE_OFFSET: tl.constexpr,
    HW: tl.constexpr,
    NUM_HW_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    spatial_pid = tl.program_id(1)
    n = spatial_pid // NUM_HW_TILES
    hw_tile = spatial_pid - n * NUM_HW_TILES
    hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_matrix = c[:, None]
    hw_matrix = hw[None, :]
    active = (c_matrix < C) & (hw_matrix < HW)
    compact_offsets = n * (C * HW) + c_matrix * HW + hw_matrix
    residual_offsets = n * (INPUT_C * HW) + (c_matrix + SLICE_OFFSET) * HW + hw_matrix

    gate = tl.load(mask_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    selected = tl.where(gate <= 0.0, fill, rhs)

    activation = tl.load(activation_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    c_mask = c < C
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    centered = _f32_sub(activation, mean)
    correction_scale = tl.load(correction_scale_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    correction = _f32_mul(centered, correction_scale)
    after_correction = _f32_sub(selected, correction)
    mean_term = tl.load(mean_term_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    after_mean = _f32_sub(after_correction, mean_term)
    output_scale = tl.load(output_scale_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    grad_bf16 = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    residual = tl.load(residual_ptr + residual_offsets, mask=active, other=0.0).to(tl.float32)
    add_value = _f32_add(residual, grad_bf16.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + compact_offsets, add_value, mask=active)


@triton.jit
def _epilogue_finalize_kernel(
    residual_ptr,
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    out_ptr,
    C: tl.constexpr,
    INPUT_C: tl.constexpr,
    SLICE_OFFSET: tl.constexpr,
    HW: tl.constexpr,
    NUM_HW_TILES: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    spatial_pid = tl.program_id(1)
    hw_tile = spatial_pid % NUM_HW_TILES
    n = spatial_pid // NUM_HW_TILES

    tile_offsets = tl.arange(0, BLOCK_TILES)
    tile_mask = tile_offsets < NUM_K_TILES
    partial_offsets = tile_offsets[:, None] * C + c[None, :]
    c_mask = c < C
    partial_mask = tile_mask[:, None] & c_mask[None, :]
    sum_value = tl.sum(
        tl.load(partial_sum_ptr + partial_offsets, mask=partial_mask, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + partial_offsets, mask=partial_mask, other=0.0).to(tl.float32),
        axis=0,
    )

    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd_sq = _f32_mul(invstd, invstd)
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    correction_scale = _f32_mul(dot_mean, invstd_sq)
    output_scale = _f32_mul(invstd, weight)
    first_tile = spatial_pid == 0
    tl.store(sum_out_ptr + c, sum_value, mask=c_mask & first_tile)
    tl.store(scaled_dot_out_ptr + c, _f32_mul(dot_value, invstd), mask=c_mask & first_tile)

    hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_matrix = c[:, None]
    hw_matrix = hw[None, :]
    active = (c_matrix < C) & (hw_matrix < HW)
    compact_offsets = n * (C * HW) + c_matrix * HW + hw_matrix
    residual_offsets = n * (INPUT_C * HW) + (c_matrix + SLICE_OFFSET) * HW + hw_matrix

    gate = tl.load(mask_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    selected = tl.where(gate <= 0.0, fill, rhs)

    activation = tl.load(activation_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    centered = _f32_sub(activation, mean)
    correction = _f32_mul(centered, correction_scale[:, None])
    after_correction = _f32_sub(selected, correction)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    after_mean = _f32_sub(after_correction, mean_term[:, None])
    grad_bf16 = _f32_mul(after_mean, output_scale[:, None]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    residual = tl.load(residual_ptr + residual_offsets, mask=active, other=0.0).to(tl.float32)
    add_value = _f32_add(residual, grad_bf16.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + compact_offsets, add_value, mask=active)


def _launch(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    EPILOGUE_C: int,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
):
    residual, mask, fill, rhs, activation, mean, invstd, weight = inputs
    n, c, h, w = mask.shape
    hw = h * w
    k_total = n * hw
    input_c = int(residual.shape[1])
    num_hw_tiles = triton.cdiv(hw, BLOCK_HW)
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    partial_sum = torch.empty_strided((num_k_tiles, c), (c, 1), device=mask.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)
    sum_out = torch.empty_strided((c,), (1,), device=mask.device, dtype=torch.float32)
    scaled_dot_out = torch.empty_strided((c,), (1,), device=mask.device, dtype=torch.float32)
    mean_term = torch.empty_strided((c,), (1,), device=mask.device, dtype=torch.float32)
    correction_scale = torch.empty_strided((c,), (1,), device=mask.device, dtype=torch.float32)
    output_scale = torch.empty_strided((c,), (1,), device=mask.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (n, c, h, w),
        (c * hw, hw, w, 1),
        device=mask.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
        mask,
        fill,
        rhs,
        activation,
        mean,
        partial_sum,
        partial_dot,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=4,
    )
    _finalize_kernel[(c,)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        sum_out,
        scaled_dot_out,
        mean_term,
        correction_scale,
        output_scale,
        C=c,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_tiles,
        SCALE_VALUE=SCALE,
        num_warps=final_warps,
        num_stages=4,
    )
    _epilogue_kernel[(triton.cdiv(c, EPILOGUE_C), n * num_hw_tiles)](
        residual,
        mask,
        fill,
        rhs,
        activation,
        mean,
        mean_term,
        correction_scale,
        output_scale,
        add_out,
        C=c,
        INPUT_C=input_c,
        SLICE_OFFSET=SLICE_START,
        HW=hw,
        NUM_HW_TILES=num_hw_tiles,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=EPILOGUE_C,
        num_warps=epilogue_warps,
        num_stages=4,
    )
    return sum_out, scaled_dot_out, add_out, add_out[:, :SLICE_START, :, :]


def _launch_fused_finalize(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    EPILOGUE_C: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    residual, mask, fill, rhs, activation, mean, invstd, weight = inputs
    n, c, h, w = mask.shape
    hw = h * w
    k_total = n * hw
    input_c = int(residual.shape[1])
    num_hw_tiles = triton.cdiv(hw, BLOCK_HW)
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    partial_sum = torch.empty_strided((num_k_tiles, c), (c, 1), device=mask.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)
    sum_out = torch.empty_strided((c,), (1,), device=mask.device, dtype=torch.float32)
    scaled_dot_out = torch.empty_strided((c,), (1,), device=mask.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (n, c, h, w),
        (c * hw, hw, w, 1),
        device=mask.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
        mask,
        fill,
        rhs,
        activation,
        mean,
        partial_sum,
        partial_dot,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=4,
    )
    _epilogue_finalize_kernel[(triton.cdiv(c, EPILOGUE_C), n * num_hw_tiles)](
        residual,
        mask,
        fill,
        rhs,
        activation,
        mean,
        partial_sum,
        partial_dot,
        invstd,
        weight,
        sum_out,
        scaled_dot_out,
        add_out,
        C=c,
        INPUT_C=input_c,
        SLICE_OFFSET=SLICE_START,
        HW=hw,
        NUM_HW_TILES=num_hw_tiles,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_tiles,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=EPILOGUE_C,
        SCALE_VALUE=SCALE,
        num_warps=epilogue_warps,
        num_stages=4,
    )
    return sum_out, scaled_dot_out, add_out, add_out[:, :SLICE_START, :, :]


@oracle_impl(hardware="B200", point="3f23a743", BLOCK_K=128, BLOCK_C=8, BLOCK_HW=256, EPILOGUE_C=EPILOGUE_BLOCK_C, reduce_warps=4, final_warps=8, epilogue_warps=4, fused_finalize=False)
@oracle_impl(hardware="B200", point="5474fc63", BLOCK_K=8192, BLOCK_C=8, BLOCK_HW=256, EPILOGUE_C=EPILOGUE_BLOCK_C, reduce_warps=16, final_warps=8, epilogue_warps=4, fused_finalize=True)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    EPILOGUE_C: int,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
    fused_finalize: bool,
):
    if fused_finalize:
        return _launch_fused_finalize(
            inputs,
            BLOCK_K=BLOCK_K,
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            EPILOGUE_C=EPILOGUE_C,
            reduce_warps=reduce_warps,
            epilogue_warps=epilogue_warps,
        )
    return _launch(
        inputs,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        EPILOGUE_C=EPILOGUE_C,
        reduce_warps=reduce_warps,
        final_warps=final_warps,
        epilogue_warps=epilogue_warps,
    )
