"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 BatchNorm-backward tail by sharing the masked bf16 `where` producer across both returned f32 channel reductions, sinking the finalized channel summaries into the dense bf16 slice-add epilogue, and reducing that materialized bf16 output for the final returned f32 vector, whereas Inductor schedules the sibling reductions, broadcast-dependent epilogue, full tensor store, and dependent output reduction as generic separated regions; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output channel-reduction plan that keeps compatible summaries live across dense and reduction consumers while preserving bf16 cast boundaries; the fix is SCHEDULER_FUSION: add a guarded BN-backward reduction/epilogue template that shares the masked producer, finalizes both channel vectors once, and sinks the dense output plus its channel reduction into the same plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 7.62939453125e-06
SLICE_START = 16


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
    centered_source_ptr,
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

    centered_source = tl.load(centered_source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered = _f32_sub(centered_source, mean[None, :])
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
    scale_ptr,
    affine_rhs_ptr,
    sum_out_ptr,
    mul8_out_ptr,
    mean_term_ptr,
    correction_scale_ptr,
    output_scale_ptr,
    C: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    mask = (tile_offsets[:, None] < NUM_K_TILES) & (c[None, :] < C)
    offsets = tile_offsets[:, None] * C + c[None, :]

    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )

    c_mask = c < C
    scale = tl.load(scale_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    affine_rhs = tl.load(affine_rhs_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    scale_sq = _f32_mul(scale, scale)
    correction_scale = _f32_mul(dot_mean, scale_sq)
    post_scale = _f32_mul(scale, affine_rhs)

    tl.store(sum_out_ptr + c, sum_value, mask=c_mask)
    tl.store(mul8_out_ptr + c, _f32_mul(dot_value, scale), mask=c_mask)
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, SCALE_VALUE), mask=c_mask)
    tl.store(correction_scale_ptr + c, correction_scale, mask=c_mask)
    tl.store(output_scale_ptr + c, post_scale, mask=c_mask)


@triton.jit
def _epilogue_and_partial_sum_kernel(
    residual_ptr,
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    centered_source_ptr,
    mean_ptr,
    mean_term_ptr,
    correction_scale_ptr,
    output_scale_ptr,
    add_out_ptr,
    partial_add_sum_ptr,
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

    centered_source = tl.load(centered_source_ptr + compact_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    c_mask = c < C
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    centered = _f32_sub(centered_source, mean)
    correction_scale = tl.load(correction_scale_ptr + c, mask=c_mask, other=0.0).to(
        tl.float32
    )[:, None]
    correction = _f32_mul(centered, correction_scale)
    after_correction = _f32_sub(selected, correction)
    mean_term = tl.load(mean_term_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    centered_grad = _f32_sub(after_correction, mean_term)
    output_scale = tl.load(output_scale_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[
        :, None
    ]
    grad = _f32_mul(centered_grad, output_scale)
    grad_bf16 = grad.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    residual = tl.load(residual_ptr + residual_offsets, mask=active, other=0.0).to(tl.float32)
    add_value = _f32_add(residual, grad_bf16)
    add_bf16 = add_value.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(add_out_ptr + compact_offsets, add_bf16, mask=active)

    add_sum = tl.sum(tl.where(active, add_bf16.to(tl.float32), 0.0), axis=1)
    partial_offsets = spatial_pid * C + c
    tl.store(partial_add_sum_ptr + partial_offsets, add_sum, mask=c_mask)


@triton.jit
def _final_add_sum_kernel(
    partial_add_sum_ptr,
    add_sum_out_ptr,
    C: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    partial = tl.arange(0, BLOCK_PARTIALS)
    mask = (partial[:, None] < NUM_PARTIALS) & (c[None, :] < C)
    offsets = partial[:, None] * C + c[None, :]
    total = tl.sum(
        tl.load(partial_add_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    tl.store(add_sum_out_ptr + c, total, mask=c < C)


@oracle_impl(
    hardware="B200",
    point="7ba0254f",
    BLOCK_K=1024,
    REDUCE_BLOCK_C=8,
    FINAL_BLOCK_C=8,
    EPILOGUE_BLOCK_HW=256,
    EPILOGUE_BLOCK_C=8,
    ADD_SUM_BLOCK_C=8,
    reduce_warps=4,
    finalize_warps=4,
    epilogue_warps=4,
    add_sum_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    REDUCE_BLOCK_C: int,
    FINAL_BLOCK_C: int,
    EPILOGUE_BLOCK_HW: int,
    EPILOGUE_BLOCK_C: int,
    ADD_SUM_BLOCK_C: int,
    reduce_warps: int,
    finalize_warps: int,
    epilogue_warps: int,
    add_sum_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    batch = int(arg1_1.shape[0])
    channels = int(arg1_1.shape[1])
    height = int(arg1_1.shape[2])
    width = int(arg1_1.shape[3])
    input_channels = int(arg0_1.shape[1])
    hw = height * width
    k_total = batch * hw
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    block_k_tiles = _next_power_of_2(num_k_tiles)
    num_hw_tiles = triton.cdiv(hw, EPILOGUE_BLOCK_HW)
    num_add_partials = batch * num_hw_tiles
    block_add_partials = _next_power_of_2(num_add_partials)

    partial_sum = torch.empty((num_k_tiles, channels), device=arg1_1.device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, channels), device=arg1_1.device, dtype=torch.float32)
    sum_out = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    mul8_out = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    mean_term = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    correction_scale = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    output_scale = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (batch, channels, height, width),
        (channels * hw, hw, width, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    partial_add_sum = torch.empty(
        (num_add_partials, channels),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    add_sum_out = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)

    _partial_reduce_kernel[(triton.cdiv(channels, REDUCE_BLOCK_C), num_k_tiles)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        partial_sum,
        partial_dot,
        C=channels,
        HW=hw,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        BLOCK_C=REDUCE_BLOCK_C,
        num_warps=reduce_warps,
    )
    _finalize_kernel[(triton.cdiv(channels, FINAL_BLOCK_C),)](
        partial_sum,
        partial_dot,
        arg6_1,
        arg7_1,
        sum_out,
        mul8_out,
        mean_term,
        correction_scale,
        output_scale,
        C=channels,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_k_tiles,
        BLOCK_C=FINAL_BLOCK_C,
        SCALE_VALUE=SCALE,
        num_warps=finalize_warps,
    )
    _epilogue_and_partial_sum_kernel[
        (triton.cdiv(channels, EPILOGUE_BLOCK_C), num_add_partials)
    ](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        mean_term,
        correction_scale,
        output_scale,
        add_out,
        partial_add_sum,
        C=channels,
        INPUT_C=input_channels,
        SLICE_OFFSET=SLICE_START,
        HW=hw,
        NUM_HW_TILES=num_hw_tiles,
        BLOCK_HW=EPILOGUE_BLOCK_HW,
        BLOCK_C=EPILOGUE_BLOCK_C,
        num_warps=epilogue_warps,
    )
    _final_add_sum_kernel[(triton.cdiv(channels, ADD_SUM_BLOCK_C),)](
        partial_add_sum,
        add_sum_out,
        C=channels,
        NUM_PARTIALS=num_add_partials,
        BLOCK_PARTIALS=block_add_partials,
        BLOCK_C=ADD_SUM_BLOCK_C,
        num_warps=add_sum_warps,
    )
    return sum_out, mul8_out, add_out, add_sum_out
