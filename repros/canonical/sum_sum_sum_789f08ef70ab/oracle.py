"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 channel-shuffle plus dual BatchNorm-backward tail by reading the two shuffled split views directly from the virtual `cat -> view -> permute -> clone` layout, sharing each branch's ReLU/where producer across the sibling per-channel reductions, and feeding finalized channel summaries into the returned channel-last bf16 gradient epilogues, whereas Inductor lowers the shuffle materialization, masked producers, four channel reductions, and dependent broadcast epilogues as separate generic schedules over materialized intermediates; Inductor cannot do this today because its scheduler/codegen does not represent fixed channel-shuffle layouts as virtual reduction inputs and cannot keep compatible BN-backward reductions together with their dependent full-tensor epilogues; the fix is SCHEDULER_FUSION: teach reduction scheduling to sink static channel-shuffle producers into multi-output channel reductions and emit the dependent BN-backward epilogues from the same planned lowering."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 0.00015943877551020407


@triton.jit
def _load_shuffled_value(
    arg0_ptr,
    arg1_ptr,
    n,
    h,
    w,
    hw,
    c_offsets,
    active_mask,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    ODD: tl.constexpr,
):
    cat_c = 2 * c_offsets[None, :] + ODD
    from_arg0 = cat_c < C
    arg0_offsets = n[:, None] * (2 * C * H * W) + cat_c * (H * W) + hw[:, None]
    arg1_c = cat_c - C
    arg1_offsets = n[:, None] * (C * H * W) + h[:, None] * (W * C) + w[:, None] * C + arg1_c
    from_arg0_value = tl.load(
        arg0_ptr + arg0_offsets,
        mask=active_mask & from_arg0,
        other=0.0,
    ).to(tl.float32)
    from_arg1_value = tl.load(
        arg1_ptr + arg1_offsets,
        mask=active_mask & ~from_arg0,
        other=0.0,
    ).to(tl.float32)
    return tl.where(from_arg0, from_arg0_value, from_arg1_value)


@triton.jit
def _partial_kernel(
    arg0_ptr,
    arg1_ptr,
    x_a_ptr,
    mean_a_ptr,
    invstd_a_ptr,
    weight_a_ptr,
    bias_a_ptr,
    zero_ptr,
    x_b_ptr,
    mean_b_ptr,
    invstd_b_ptr,
    weight_b_ptr,
    bias_b_ptr,
    partials_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL_SPATIAL: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    hw = k_offsets % (H * W)
    n = k_offsets // (H * W)
    h = hw // W
    w = hw - h * W
    c_mask = c_offsets < C
    mask = (k_offsets[:, None] < TOTAL_SPATIAL) & c_mask[None, :]
    offsets = n[:, None] * C * H * W + h[:, None] * W * C + w[:, None] * C + c_offsets[None, :]

    zero = tl.load(zero_ptr).to(tl.float32)

    x_a = tl.load(x_a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_a = tl.load(mean_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_a = tl.load(invstd_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_a = tl.load(weight_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias_a = tl.load(bias_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered_a = x_a - mean_a[None, :]
    affine_a = centered_a * invstd_a[None, :]
    affine_a = affine_a * weight_a[None, :] + bias_a[None, :]
    affine_a_bf16 = affine_a.to(tl.bfloat16)
    odd_shuffle = _load_shuffled_value(
        arg0_ptr,
        arg1_ptr,
        n,
        h,
        w,
        hw,
        c_offsets,
        mask,
        C=C,
        H=H,
        W=W,
        ODD=1,
    )
    where_a = tl.where(affine_a_bf16 <= 0.0, zero, odd_shuffle)
    where_a = tl.where(mask, where_a, 0.0)

    x_b = tl.load(x_b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_b = tl.load(mean_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_b = tl.load(invstd_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_b = tl.load(weight_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias_b = tl.load(bias_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered_b = x_b - mean_b[None, :]
    affine_b = centered_b * invstd_b[None, :]
    affine_b = affine_b * weight_b[None, :] + bias_b[None, :]
    affine_b_bf16 = affine_b.to(tl.bfloat16)
    even_shuffle = _load_shuffled_value(
        arg0_ptr,
        arg1_ptr,
        n,
        h,
        w,
        hw,
        c_offsets,
        mask,
        C=C,
        H=H,
        W=W,
        ODD=0,
    )
    where_b = tl.where(affine_b_bf16 <= 0.0, zero, even_shuffle)
    where_b = tl.where(mask, where_b, 0.0)

    partial_sum_a = tl.sum(where_a, axis=0)
    partial_dot_a = tl.sum(where_a * tl.where(mask, centered_a, 0.0), axis=0)
    partial_sum_b = tl.sum(where_b, axis=0)
    partial_dot_b = tl.sum(where_b * tl.where(mask, centered_b, 0.0), axis=0)

    partial_offset = c_offsets * NUM_TILES + tl.program_id(1)
    plane = C * NUM_TILES
    tl.store(partials_ptr + partial_offset, partial_sum_a, mask=c_mask)
    tl.store(partials_ptr + plane + partial_offset, partial_dot_a, mask=c_mask)
    tl.store(partials_ptr + 2 * plane + partial_offset, partial_sum_b, mask=c_mask)
    tl.store(partials_ptr + 3 * plane + partial_offset, partial_dot_b, mask=c_mask)


@triton.jit
def _finalize_kernel(
    partials_ptr,
    invstd_a_ptr,
    weight_a_ptr,
    invstd_b_ptr,
    weight_b_ptr,
    sum_a_ptr,
    vec_a_ptr,
    sum_b_ptr,
    vec_b_ptr,
    stats_ptr,
    C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    c_mask = c_offsets < C
    tile_mask = tile_offsets < NUM_TILES
    offsets = c_offsets[None, :] * NUM_TILES + tile_offsets[:, None]
    mask = tile_mask[:, None] & c_mask[None, :]
    plane = C * NUM_TILES

    sum_a = tl.sum(tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    dot_a = tl.sum(tl.load(partials_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    sum_b = tl.sum(tl.load(partials_ptr + 2 * plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    dot_b = tl.sum(tl.load(partials_ptr + 3 * plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

    invstd_a = tl.load(invstd_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_a = tl.load(weight_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_b = tl.load(invstd_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_b = tl.load(weight_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(sum_a_ptr + c_offsets, sum_a, mask=c_mask)
    tl.store(vec_a_ptr + c_offsets, dot_a * invstd_a, mask=c_mask)
    tl.store(sum_b_ptr + c_offsets, sum_b, mask=c_mask)
    tl.store(vec_b_ptr + c_offsets, dot_b * invstd_b, mask=c_mask)
    tl.store(stats_ptr + c_offsets, sum_a * SCALE_VALUE, mask=c_mask)
    tl.store(stats_ptr + C + c_offsets, (dot_a * SCALE_VALUE) * (invstd_a * invstd_a), mask=c_mask)
    tl.store(stats_ptr + 2 * C + c_offsets, invstd_a * weight_a, mask=c_mask)
    tl.store(stats_ptr + 3 * C + c_offsets, sum_b * SCALE_VALUE, mask=c_mask)
    tl.store(stats_ptr + 4 * C + c_offsets, (dot_b * SCALE_VALUE) * (invstd_b * invstd_b), mask=c_mask)
    tl.store(stats_ptr + 5 * C + c_offsets, invstd_b * weight_b, mask=c_mask)


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    x_a_ptr,
    mean_a_ptr,
    invstd_a_ptr,
    weight_a_ptr,
    bias_a_ptr,
    zero_ptr,
    x_b_ptr,
    mean_b_ptr,
    invstd_b_ptr,
    weight_b_ptr,
    bias_b_ptr,
    stats_ptr,
    out_a_ptr,
    out_b_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    hw_offsets = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_offsets = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    h = hw_offsets // W
    w = hw_offsets - h * W
    hw_mask = hw_offsets < H * W
    c_mask = c_offsets < C
    mask = hw_mask[:, None] & c_mask[None, :]
    offsets = n * C * H * W + h[:, None] * W * C + w[:, None] * C + c_offsets[None, :]

    zero = tl.load(zero_ptr).to(tl.float32)

    x_a = tl.load(x_a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_a = tl.load(mean_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_a = tl.load(invstd_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_a = tl.load(weight_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias_a = tl.load(bias_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered_a = x_a - mean_a[None, :]
    affine_a = centered_a * invstd_a[None, :]
    affine_a = affine_a * weight_a[None, :] + bias_a[None, :]
    odd_shuffle = _load_shuffled_value(
        arg0_ptr,
        arg1_ptr,
        tl.full((BLOCK_HW,), n, tl.int64),
        h,
        w,
        hw_offsets,
        c_offsets,
        mask,
        C=C,
        H=H,
        W=W,
        ODD=1,
    )
    where_a = tl.where(affine_a.to(tl.bfloat16) <= 0.0, zero, odd_shuffle)

    x_b = tl.load(x_b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_b = tl.load(mean_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_b = tl.load(invstd_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_b = tl.load(weight_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias_b = tl.load(bias_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered_b = x_b - mean_b[None, :]
    affine_b = centered_b * invstd_b[None, :]
    affine_b = affine_b * weight_b[None, :] + bias_b[None, :]
    even_shuffle = _load_shuffled_value(
        arg0_ptr,
        arg1_ptr,
        tl.full((BLOCK_HW,), n, tl.int64),
        h,
        w,
        hw_offsets,
        c_offsets,
        mask,
        C=C,
        H=H,
        W=W,
        ODD=0,
    )
    where_b = tl.where(affine_b.to(tl.bfloat16) <= 0.0, zero, even_shuffle)

    mean_term_a = tl.load(stats_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    coeff_a = tl.load(stats_ptr + C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    scale_a = tl.load(stats_ptr + 2 * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    mean_term_b = tl.load(stats_ptr + 3 * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    coeff_b = tl.load(stats_ptr + 4 * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    scale_b = tl.load(stats_ptr + 5 * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    grad_a = ((where_a - centered_a * coeff_a[None, :]) - mean_term_a[None, :]) * scale_a[None, :]
    grad_b = ((where_b - centered_b * coeff_b[None, :]) - mean_term_b[None, :]) * scale_b[None, :]
    tl.store(out_a_ptr + offsets, grad_a.to(tl.bfloat16), mask=mask)
    tl.store(out_b_ptr + offsets, grad_b.to(tl.bfloat16), mask=mask)


@oracle_impl(
    hardware="B200",
    point="c8e869b1",
    REDUCE_BLOCK=512,
    REDUCE_BLOCK_C=8,
    FINAL_BLOCK_C=16,
    EPILOGUE_BLOCK_HW=64,
    EPILOGUE_BLOCK_C=32,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    REDUCE_BLOCK: int,
    REDUCE_BLOCK_C: int,
    FINAL_BLOCK_C: int,
    EPILOGUE_BLOCK_HW: int,
    EPILOGUE_BLOCK_C: int,
    num_warps: int,
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
        _shape_param_1,
    ) = inputs
    del _shape_param_0, _shape_param_1

    n = int(arg2_1.shape[0])
    c = int(arg2_1.shape[1])
    h = int(arg2_1.shape[2])
    w = int(arg2_1.shape[3])
    total_spatial = n * h * w
    num_tiles = triton.cdiv(total_spatial, REDUCE_BLOCK)

    partials = torch.empty_strided(
        (4, c, num_tiles),
        (c * num_tiles, num_tiles, 1),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided((c,), (1,), device=arg2_1.device, dtype=torch.float32)
    mul_10 = torch.empty_strided((c,), (1,), device=arg2_1.device, dtype=torch.float32)
    sum_3 = torch.empty_strided((c,), (1,), device=arg2_1.device, dtype=torch.float32)
    mul_21 = torch.empty_strided((c,), (1,), device=arg2_1.device, dtype=torch.float32)
    stats = torch.empty_strided((6, c), (c, 1), device=arg2_1.device, dtype=torch.float32)
    convert_element_type_3 = torch.empty_strided(
        (n, c, h, w),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    convert_element_type_7 = torch.empty_strided(
        (n, c, h, w),
        tuple(arg8_1.stride()),
        device=arg8_1.device,
        dtype=torch.bfloat16,
    )

    _partial_kernel[(triton.cdiv(c, REDUCE_BLOCK_C), num_tiles)](
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
        partials,
        C=c,
        H=h,
        W=w,
        TOTAL_SPATIAL=total_spatial,
        NUM_TILES=num_tiles,
        BLOCK_K=REDUCE_BLOCK,
        BLOCK_C=REDUCE_BLOCK_C,
        num_warps=num_warps,
    )
    _finalize_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        partials,
        arg4_1,
        arg5_1,
        arg10_1,
        arg11_1,
        sum_1,
        mul_10,
        sum_3,
        mul_21,
        stats,
        C=c,
        NUM_TILES=num_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        BLOCK_C=FINAL_BLOCK_C,
        SCALE_VALUE=SCALE,
        num_warps=1,
    )
    _epilogue_kernel[(n, triton.cdiv(h * w, EPILOGUE_BLOCK_HW), triton.cdiv(c, EPILOGUE_BLOCK_C))](
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
        stats,
        convert_element_type_3,
        convert_element_type_7,
        C=c,
        H=h,
        W=w,
        BLOCK_HW=EPILOGUE_BLOCK_HW,
        BLOCK_C=EPILOGUE_BLOCK_C,
        num_warps=4,
    )

    return sum_1, mul_10, convert_element_type_3, sum_3, mul_21, convert_element_type_7
