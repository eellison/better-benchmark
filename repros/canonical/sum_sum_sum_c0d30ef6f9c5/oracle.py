"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete RepVGG bf16 dual-BatchNorm-backward scope by replacing the zero-fill/as_strided_scatter/as_strided/expand/div chain with its exact observed bf16 `arg0 / 49` broadcast, sharing the single masked producer and duplicate sum output across both channel-reduction branches, and feeding finalized per-channel scalars directly into both returned channels-last dense-gradient epilogues, whereas Inductor lowers the scatter-style broadcast construction, bf16 affine/add/ReLU mask producer, duplicate reductions, and dependent epilogues as generic scheduled regions; Inductor cannot do this today because its algebraic simplifier does not canonicalize this scatter-to-broadcast idiom or deduplicate the identical reduction before reduction scheduling; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to collapse zero-fill scatter broadcasts and common returned reductions while preserving bf16 cast boundaries for downstream BN-backward epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 0.00015943877551020407


@triton.jit
def _partial_kernel(
    arg0_ptr,
    arg1_ptr,
    mean_a_ptr,
    invstd_a_ptr,
    weight_a_ptr,
    bias_a_ptr,
    arg6_ptr,
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
    nhw_offsets = n[:, None] * C * H * W + h[:, None] * W * C + w[:, None] * C + c_offsets[None, :]

    x_a = tl.load(arg1_ptr + nhw_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_a = tl.load(mean_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_a = tl.load(invstd_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_a = tl.load(weight_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias_a = tl.load(bias_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered_a = x_a - mean_a[None, :]
    branch_a = centered_a * invstd_a[None, :]
    branch_a = branch_a * weight_a[None, :] + bias_a[None, :]
    branch_a_bf16 = branch_a.to(tl.bfloat16)

    x_b = tl.load(arg6_ptr + nhw_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_b = tl.load(mean_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_b = tl.load(invstd_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_b = tl.load(weight_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias_b = tl.load(bias_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered_b = x_b - mean_b[None, :]
    branch_b = centered_b * invstd_b[None, :]
    branch_b = branch_b * weight_b[None, :] + bias_b[None, :]
    branch_b_bf16 = branch_b.to(tl.bfloat16)

    add_bf16 = (branch_a_bf16 + branch_b_bf16).to(tl.bfloat16)
    broadcast = tl.load(arg0_ptr + n[:, None] * C + c_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
    broadcast = (broadcast / 49.0).to(tl.bfloat16).to(tl.float32)
    where_value = tl.where(add_bf16 <= 0.0, 0.0, broadcast)
    where_value = tl.where(mask, where_value, 0.0)

    partial_sum = tl.sum(where_value, axis=0)
    partial_dot_b = tl.sum(where_value * tl.where(mask, centered_b, 0.0), axis=0)
    partial_dot_a = tl.sum(where_value * tl.where(mask, centered_a, 0.0), axis=0)

    partial_offset = c_offsets * NUM_TILES + tl.program_id(1)
    plane = C * NUM_TILES
    tl.store(partials_ptr + partial_offset, partial_sum, mask=c_mask)
    tl.store(partials_ptr + plane + partial_offset, partial_dot_b, mask=c_mask)
    tl.store(partials_ptr + 2 * plane + partial_offset, partial_dot_a, mask=c_mask)


@triton.jit
def _finalize_kernel(
    partials_ptr,
    invstd_b_ptr,
    weight_b_ptr,
    invstd_a_ptr,
    weight_a_ptr,
    scalar_ptr,
    sum1_ptr,
    vec_b_ptr,
    sum3_ptr,
    vec_a_ptr,
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

    sum_value = tl.sum(tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    dot_b = tl.sum(tl.load(partials_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    dot_a = tl.sum(tl.load(partials_ptr + 2 * plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

    invstd_b = tl.load(invstd_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_b = tl.load(weight_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_a = tl.load(invstd_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_a = tl.load(weight_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(sum1_ptr + c_offsets, sum_value, mask=c_mask)
    tl.store(sum3_ptr + c_offsets, sum_value, mask=c_mask)
    tl.store(vec_b_ptr + c_offsets, dot_b * invstd_b, mask=c_mask)
    tl.store(vec_a_ptr + c_offsets, dot_a * invstd_a, mask=c_mask)
    tl.store(stats_ptr + c_offsets, sum_value * SCALE_VALUE, mask=c_mask)
    tl.store(stats_ptr + C + c_offsets, (dot_b * SCALE_VALUE) * (invstd_b * invstd_b), mask=c_mask)
    tl.store(stats_ptr + 2 * C + c_offsets, invstd_b * weight_b, mask=c_mask)
    tl.store(stats_ptr + 3 * C + c_offsets, (dot_a * SCALE_VALUE) * (invstd_a * invstd_a), mask=c_mask)
    tl.store(stats_ptr + 4 * C + c_offsets, invstd_a * weight_a, mask=c_mask)

    if tl.program_id(0) == 0:
        tl.store(scalar_ptr, 0.0)


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    mean_a_ptr,
    invstd_a_ptr,
    weight_a_ptr,
    bias_a_ptr,
    arg6_ptr,
    mean_b_ptr,
    invstd_b_ptr,
    weight_b_ptr,
    bias_b_ptr,
    stats_ptr,
    out_b_ptr,
    out_a_ptr,
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

    x_a = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_a = tl.load(mean_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_a = tl.load(invstd_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_a = tl.load(weight_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias_a = tl.load(bias_a_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered_a = x_a - mean_a[None, :]
    branch_a = centered_a * invstd_a[None, :]
    branch_a = branch_a * weight_a[None, :] + bias_a[None, :]
    branch_a_bf16 = branch_a.to(tl.bfloat16)

    x_b = tl.load(arg6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_b = tl.load(mean_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd_b = tl.load(invstd_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight_b = tl.load(weight_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias_b = tl.load(bias_b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered_b = x_b - mean_b[None, :]
    branch_b = centered_b * invstd_b[None, :]
    branch_b = branch_b * weight_b[None, :] + bias_b[None, :]
    branch_b_bf16 = branch_b.to(tl.bfloat16)

    add_bf16 = (branch_a_bf16 + branch_b_bf16).to(tl.bfloat16)
    broadcast = tl.load(arg0_ptr + n * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    broadcast = (broadcast / 49.0).to(tl.bfloat16).to(tl.float32)
    where_value = tl.where(add_bf16 <= 0.0, 0.0, broadcast[None, :])

    mean_term = tl.load(stats_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    coeff_b = tl.load(stats_ptr + C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    scale_b = tl.load(stats_ptr + 2 * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    coeff_a = tl.load(stats_ptr + 3 * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    scale_a = tl.load(stats_ptr + 4 * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    out_b = ((where_value - centered_b * coeff_b[None, :]) - mean_term[None, :]) * scale_b[None, :]
    out_a = ((where_value - centered_a * coeff_a[None, :]) - mean_term[None, :]) * scale_a[None, :]
    tl.store(out_b_ptr + offsets, out_b.to(tl.bfloat16), mask=mask)
    tl.store(out_a_ptr + offsets, out_a.to(tl.bfloat16), mask=mask)


@oracle_impl(
    hardware="B200",
    point="ac965caa",
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
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    ) = inputs
    n = int(arg1_1.shape[0])
    c = int(arg1_1.shape[1])
    h = int(arg1_1.shape[2])
    w = int(arg1_1.shape[3])
    total_spatial = n * h * w
    num_tiles = triton.cdiv(total_spatial, REDUCE_BLOCK)

    scalar_zero = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.bfloat16)
    partials = torch.empty_strided(
        (3, c, num_tiles),
        (c * num_tiles, num_tiles, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided((c,), (1,), device=arg0_1.device, dtype=torch.float32)
    mul_12 = torch.empty_strided((c,), (1,), device=arg0_1.device, dtype=torch.float32)
    sum_3 = torch.empty_strided((c,), (1,), device=arg0_1.device, dtype=torch.float32)
    mul_21 = torch.empty_strided((c,), (1,), device=arg0_1.device, dtype=torch.float32)
    stats = torch.empty_strided((5, c), (c, 1), device=arg0_1.device, dtype=torch.float32)
    grad_b = torch.empty_strided(
        (n, c, h, w),
        tuple(arg6_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grad_a = torch.empty_strided(
        (n, c, h, w),
        tuple(arg1_1.stride()),
        device=arg0_1.device,
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
        arg8_1,
        arg9_1,
        arg3_1,
        arg4_1,
        scalar_zero,
        sum_1,
        mul_12,
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
        stats,
        grad_b,
        grad_a,
        C=c,
        H=h,
        W=w,
        BLOCK_HW=EPILOGUE_BLOCK_HW,
        BLOCK_C=EPILOGUE_BLOCK_C,
        num_warps=4,
    )
    return scalar_zero, sum_1, mul_12, grad_b, sum_3, mul_21, grad_a
