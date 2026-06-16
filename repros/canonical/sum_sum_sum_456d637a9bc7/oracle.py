"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 channel-shuffle plus dual BatchNorm-backward scope by keeping the static slice/cat/view/permute/clone producer virtual, sharing one split-K pass for the two masked per-channel sum pairs, and feeding the finalized channel scalars directly into both returned channels-last dense-gradient epilogues. Inductor lowers the shuffle, bf16 ReLU-mask/where producers, sibling reductions, and dependent dense epilogues as generic scheduled regions with avoidable materialization and replay. The fix is SCHEDULER_FUSION: teach reduction scheduling to inline static ShuffleNet channel-shuffle producers while planning paired channel reductions and BN-backward epilogues in one fused template."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 9.964923469387754e-06


@triton.jit
def _partial_two_branch_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    mean1_ptr,
    invstd1_ptr,
    weight1_ptr,
    bias1_ptr,
    fill_ptr,
    arg8_ptr,
    mean2_ptr,
    invstd2_ptr,
    weight2_ptr,
    bias2_ptr,
    partials_ptr,
    C: tl.constexpr,
    SHUFFLED_C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL_SPATIAL: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tile = tl.program_id(1)
    k = tile * BLOCK + tl.arange(0, BLOCK)
    active = k[:, None] < TOTAL_SPATIAL
    hw = k % (H * W)
    n = k // (H * W)
    h = hw // W
    w = hw - h * W
    c_active = c_offsets < C
    mask = active & c_active[None, :]

    nhwc_offsets = (
        n[:, None] * C * H * W
        + h[:, None] * W * C
        + w[:, None] * C
        + c_offsets[None, :]
    )
    fill = tl.load(fill_ptr).to(tl.float32)

    source_odd = c_offsets * 2 + 1
    odd_from_arg0 = source_odd < C
    odd_arg0_offsets = (
        n[:, None] * SHUFFLED_C * H * W
        + source_odd[None, :] * H * W
        + h[:, None] * W
        + w[:, None]
    )
    odd_arg1_c = source_odd - C
    odd_arg1_offsets = (
        n[:, None] * C * H * W
        + h[:, None] * W * C
        + w[:, None] * C
        + odd_arg1_c[None, :]
    )
    odd_arg0 = tl.load(
        arg0_ptr + odd_arg0_offsets,
        mask=mask & odd_from_arg0[None, :],
        other=0.0,
    )
    odd_arg1 = tl.load(
        arg1_ptr + odd_arg1_offsets,
        mask=mask & (odd_from_arg0[None, :] == 0),
        other=0.0,
    )
    shuffled_odd = tl.where(odd_from_arg0[None, :], odd_arg0, odd_arg1).to(tl.float32)

    x1 = tl.load(arg2_ptr + nhwc_offsets, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
    invstd1 = tl.load(invstd1_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
    centered1 = x1 - mean1[None, :]
    affine1 = centered1 * invstd1[None, :]
    affine1 = affine1 * weight1[None, :] + bias1[None, :]
    where1 = tl.where(affine1.to(tl.bfloat16) <= 0.0, fill, shuffled_odd)
    partial_sum1 = tl.sum(tl.where(mask, where1, 0.0), axis=0)
    partial_dot1 = tl.sum(tl.where(mask, where1 * centered1, 0.0), axis=0)

    source_even = c_offsets * 2
    even_from_arg0 = source_even < C
    even_arg0_offsets = (
        n[:, None] * SHUFFLED_C * H * W
        + source_even[None, :] * H * W
        + h[:, None] * W
        + w[:, None]
    )
    even_arg1_c = source_even - C
    even_arg1_offsets = (
        n[:, None] * C * H * W
        + h[:, None] * W * C
        + w[:, None] * C
        + even_arg1_c[None, :]
    )
    even_arg0 = tl.load(
        arg0_ptr + even_arg0_offsets,
        mask=mask & even_from_arg0[None, :],
        other=0.0,
    )
    even_arg1 = tl.load(
        arg1_ptr + even_arg1_offsets,
        mask=mask & (even_from_arg0[None, :] == 0),
        other=0.0,
    )
    shuffled_even = tl.where(even_from_arg0[None, :], even_arg0, even_arg1).to(tl.float32)

    x2 = tl.load(arg8_ptr + nhwc_offsets, mask=mask, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
    invstd2 = tl.load(invstd2_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
    centered2 = x2 - mean2[None, :]
    affine2 = centered2 * invstd2[None, :]
    affine2 = affine2 * weight2[None, :] + bias2[None, :]
    where2 = tl.where(affine2.to(tl.bfloat16) <= 0.0, fill, shuffled_even)
    partial_sum2 = tl.sum(tl.where(mask, where2, 0.0), axis=0)
    partial_dot2 = tl.sum(tl.where(mask, where2 * centered2, 0.0), axis=0)

    partial_offset = c_offsets * NUM_TILES + tile
    plane = C * NUM_TILES
    tl.store(partials_ptr + partial_offset, partial_sum1, mask=c_active)
    tl.store(partials_ptr + plane + partial_offset, partial_dot1, mask=c_active)
    tl.store(partials_ptr + 2 * plane + partial_offset, partial_sum2, mask=c_active)
    tl.store(partials_ptr + 3 * plane + partial_offset, partial_dot2, mask=c_active)


@triton.jit
def _finalize_two_branch_kernel(
    partials_ptr,
    invstd1_ptr,
    invstd2_ptr,
    sum1_ptr,
    vec1_ptr,
    sum2_ptr,
    vec2_ptr,
    stats_ptr,
    C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < NUM_TILES
    offsets = c * NUM_TILES + tiles
    plane = C * NUM_TILES

    sum1 = tl.sum(tl.load(partials_ptr + offsets, mask=active, other=0.0).to(tl.float32), axis=0)
    dot1 = tl.sum(tl.load(partials_ptr + plane + offsets, mask=active, other=0.0).to(tl.float32), axis=0)
    sum2 = tl.sum(tl.load(partials_ptr + 2 * plane + offsets, mask=active, other=0.0).to(tl.float32), axis=0)
    dot2 = tl.sum(tl.load(partials_ptr + 3 * plane + offsets, mask=active, other=0.0).to(tl.float32), axis=0)

    invstd1 = tl.load(invstd1_ptr + c).to(tl.float32)
    invstd2 = tl.load(invstd2_ptr + c).to(tl.float32)

    tl.store(sum1_ptr + c, sum1)
    tl.store(vec1_ptr + c, dot1 * invstd1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(vec2_ptr + c, dot2 * invstd2)
    tl.store(stats_ptr + c, sum1 * SCALE_VALUE)
    tl.store(stats_ptr + C + c, (dot1 * SCALE_VALUE) * (invstd1 * invstd1))
    tl.store(stats_ptr + 2 * C + c, sum2 * SCALE_VALUE)
    tl.store(stats_ptr + 3 * C + c, (dot2 * SCALE_VALUE) * (invstd2 * invstd2))


@triton.jit
def _epilogue_two_branch_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    mean1_ptr,
    invstd1_ptr,
    weight1_ptr,
    bias1_ptr,
    fill_ptr,
    arg8_ptr,
    mean2_ptr,
    invstd2_ptr,
    weight2_ptr,
    bias2_ptr,
    stats_ptr,
    out1_ptr,
    out2_ptr,
    NUMEL: tl.constexpr,
    C: tl.constexpr,
    SHUFFLED_C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < NUMEL
    c = offsets % C
    tmp = offsets // C
    w = tmp % W
    tmp = tmp // W
    h = tmp % H
    n = tmp // H
    fill = tl.load(fill_ptr).to(tl.float32)

    source_odd = c * 2 + 1
    odd_from_arg0 = source_odd < C
    odd_arg0_offsets = n * SHUFFLED_C * H * W + source_odd * H * W + h * W + w
    odd_arg1_c = source_odd - C
    odd_arg1_offsets = n * C * H * W + h * W * C + w * C + odd_arg1_c
    odd_arg0 = tl.load(arg0_ptr + odd_arg0_offsets, mask=active & odd_from_arg0, other=0.0)
    odd_arg1 = tl.load(arg1_ptr + odd_arg1_offsets, mask=active & (odd_from_arg0 == 0), other=0.0)
    shuffled_odd = tl.where(odd_from_arg0, odd_arg0, odd_arg1).to(tl.float32)

    x1 = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd1 = tl.load(invstd1_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered1 = x1 - mean1
    affine1 = centered1 * invstd1
    affine1 = affine1 * weight1 + bias1
    where1 = tl.where(affine1.to(tl.bfloat16) <= 0.0, fill, shuffled_odd)
    mean_term1 = tl.load(stats_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff1 = tl.load(stats_ptr + C + c, mask=active, other=0.0).to(tl.float32)
    out1 = ((where1 - centered1 * coeff1) - mean_term1) * (invstd1 * weight1)

    source_even = c * 2
    even_from_arg0 = source_even < C
    even_arg0_offsets = n * SHUFFLED_C * H * W + source_even * H * W + h * W + w
    even_arg1_c = source_even - C
    even_arg1_offsets = n * C * H * W + h * W * C + w * C + even_arg1_c
    even_arg0 = tl.load(arg0_ptr + even_arg0_offsets, mask=active & even_from_arg0, other=0.0)
    even_arg1 = tl.load(arg1_ptr + even_arg1_offsets, mask=active & (even_from_arg0 == 0), other=0.0)
    shuffled_even = tl.where(even_from_arg0, even_arg0, even_arg1).to(tl.float32)

    x2 = tl.load(arg8_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd2 = tl.load(invstd2_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered2 = x2 - mean2
    affine2 = centered2 * invstd2
    affine2 = affine2 * weight2 + bias2
    where2 = tl.where(affine2.to(tl.bfloat16) <= 0.0, fill, shuffled_even)
    mean_term2 = tl.load(stats_ptr + 2 * C + c, mask=active, other=0.0).to(tl.float32)
    coeff2 = tl.load(stats_ptr + 3 * C + c, mask=active, other=0.0).to(tl.float32)
    out2 = ((where2 - centered2 * coeff2) - mean_term2) * (invstd2 * weight2)

    tl.store(out1_ptr + offsets, out1.to(tl.bfloat16), mask=active)
    tl.store(out2_ptr + offsets, out2.to(tl.bfloat16), mask=active)


@oracle_impl(
    hardware="B200",
    point="3df80ba2",
    REDUCE_BLOCK=1024,
    BLOCK_C=4,
    EPILOGUE_BLOCK=512,
    num_warps=4,
)
def oracle_forward(inputs, *, REDUCE_BLOCK: int, BLOCK_C: int, EPILOGUE_BLOCK: int, num_warps: int):
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
    shuffled_c = int(arg0_1.shape[1])
    total_spatial = n * h * w
    grad_numel = n * c * h * w
    num_tiles = triton.cdiv(total_spatial, REDUCE_BLOCK)

    partials = torch.empty_strided(
        (4, c, num_tiles),
        (c * num_tiles, num_tiles, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum1 = torch.empty_strided((c,), (1,), device=arg0_1.device, dtype=torch.float32)
    vec1 = torch.empty_strided((c,), (1,), device=arg0_1.device, dtype=torch.float32)
    sum2 = torch.empty_strided((c,), (1,), device=arg0_1.device, dtype=torch.float32)
    vec2 = torch.empty_strided((c,), (1,), device=arg0_1.device, dtype=torch.float32)
    stats = torch.empty_strided((4, c), (c, 1), device=arg0_1.device, dtype=torch.float32)
    grad1 = torch.empty_strided(
        (n, c, h, w),
        tuple(int(stride) for stride in arg2_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grad2 = torch.empty_strided(
        (n, c, h, w),
        tuple(int(stride) for stride in arg8_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _partial_two_branch_kernel[(triton.cdiv(c, BLOCK_C), num_tiles)](
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
        SHUFFLED_C=shuffled_c,
        H=h,
        W=w,
        TOTAL_SPATIAL=total_spatial,
        NUM_TILES=num_tiles,
        BLOCK=REDUCE_BLOCK,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    _finalize_two_branch_kernel[(c,)](
        partials,
        arg4_1,
        arg10_1,
        sum1,
        vec1,
        sum2,
        vec2,
        stats,
        C=c,
        NUM_TILES=num_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        SCALE_VALUE=SCALE,
        num_warps=1,
    )
    _epilogue_two_branch_kernel[(triton.cdiv(grad_numel, EPILOGUE_BLOCK),)](
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
        grad1,
        grad2,
        NUMEL=grad_numel,
        C=c,
        SHUFFLED_C=shuffled_c,
        H=h,
        W=w,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=num_warps,
    )
    return sum1, vec1, grad1, sum2, vec2, grad2
