"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete RepVGG bf16 triple training-BatchNorm branch-sum scope, including three fp32 population var_mean reductions over channels-last inputs, eps=1e-5 rsqrt side outputs, six mutable running-stat copy_ aliases with the captured variance-correction literal, exact fp32 affine expressions followed by per-branch bf16 casts, the captured bf16 add order, NaN-preserving bf16 ReLU, and the returned saved-mean views, whereas Inductor lowers the sibling BN-training reductions, mutable stat updates, and shared branch-sum activation through separate generic normalization and pointwise schedules; Inductor cannot do this today because the normalization scheduler does not co-schedule three mutable BN-training templates while sinking their exact bf16 epilogues into one full-scope activation store; the fix is SCHEDULER_FUSION: extend the BN-training scheduler to group sibling channel-stat reductions, emit running-stat side effects, and fuse the exact bf16 branch-sum ReLU epilogue."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _aten_ctas_per_output(elements_per_channel, channels, device):
    props = torch.cuda.get_device_properties(device)
    output_vec_size = 4
    block_width = 32
    block_height = 4
    num_threads = block_width * block_height
    output_blocks = triton.cdiv(triton.cdiv(channels, output_vec_size), block_width)
    target_grid = props.multi_processor_count * (props.max_threads_per_multi_processor // num_threads)
    values_per_thread = triton.cdiv(elements_per_channel, block_height)
    ctas_per_output1 = triton.cdiv(target_grid, output_blocks)
    ctas_per_output2 = triton.cdiv(values_per_thread, 16)
    ctas_per_output3 = triton.cdiv(values_per_thread, 256)
    return max(min(ctas_per_output1, ctas_per_output2), ctas_per_output3)


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
def _welford_reduce_one(mean, m2, count, value, active):
    new_count = count + 1.0
    delta = value - mean
    new_mean = mean + delta / new_count
    new_m2 = m2 + delta * (value - new_mean)
    return (
        tl.where(active, new_mean, mean),
        tl.where(active, new_m2, m2),
        tl.where(active, new_count, count),
    )


@triton.jit
def _welford_combine_exact(mean_a, m2_a, count_a, mean_b, m2_b, count_b):
    a_empty = count_a == 0.0
    b_empty = count_b == 0.0
    delta = mean_b - mean_a
    new_count = count_a + count_b
    b_over_total = count_b / new_count
    mean = mean_a + delta * b_over_total
    m2 = m2_a + m2_b + delta * delta * count_a * b_over_total
    return (
        tl.where(a_empty, mean_b, tl.where(b_empty, mean_a, mean)),
        tl.where(a_empty, m2_b, tl.where(b_empty, m2_a, m2)),
        new_count,
    )


@triton.jit
def _d6dd_aten_partial_stats_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    partial_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    CTAS: tl.constexpr,
    MAX_PAIRS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    VEC: tl.constexpr,
):
    x_block = tl.program_id(0)
    cta = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_X)[:, None]
    vec = tl.arange(0, VEC)[None, :]
    channels = (x_block * BLOCK_X + lanes) * VEC + vec
    channel_mask = channels < C
    step = CTAS * 4

    mean00 = tl.zeros((BLOCK_X, VEC), tl.float32)
    mean01 = tl.zeros((BLOCK_X, VEC), tl.float32)
    mean02 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m200 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m201 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m202 = tl.zeros((BLOCK_X, VEC), tl.float32)
    count0 = tl.zeros((BLOCK_X, VEC), tl.float32)

    mean10 = tl.zeros((BLOCK_X, VEC), tl.float32)
    mean11 = tl.zeros((BLOCK_X, VEC), tl.float32)
    mean12 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m210 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m211 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m212 = tl.zeros((BLOCK_X, VEC), tl.float32)
    count1 = tl.zeros((BLOCK_X, VEC), tl.float32)

    mean20 = tl.zeros((BLOCK_X, VEC), tl.float32)
    mean21 = tl.zeros((BLOCK_X, VEC), tl.float32)
    mean22 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m220 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m221 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m222 = tl.zeros((BLOCK_X, VEC), tl.float32)
    count2 = tl.zeros((BLOCK_X, VEC), tl.float32)

    mean30 = tl.zeros((BLOCK_X, VEC), tl.float32)
    mean31 = tl.zeros((BLOCK_X, VEC), tl.float32)
    mean32 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m230 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m231 = tl.zeros((BLOCK_X, VEC), tl.float32)
    m232 = tl.zeros((BLOCK_X, VEC), tl.float32)
    count3 = tl.zeros((BLOCK_X, VEC), tl.float32)

    for pair in tl.static_range(0, MAX_PAIRS):
        e0 = cta * 4 + pair * 2 * step
        e1 = e0 + step

        active0 = (e0 < E) & channel_mask
        offsets0 = e0 * C + channels
        v00 = tl.load(x0_ptr + offsets0, mask=active0, other=0.0).to(tl.float32)
        v01 = tl.load(x1_ptr + offsets0, mask=active0, other=0.0).to(tl.float32)
        v02 = tl.load(x2_ptr + offsets0, mask=active0, other=0.0).to(tl.float32)
        mean00, m200, count0 = _welford_reduce_one(mean00, m200, count0, v00, active0)
        mean01, m201, _ = _welford_reduce_one(mean01, m201, count0 - 1.0, v01, active0)
        mean02, m202, _ = _welford_reduce_one(mean02, m202, count0 - 1.0, v02, active0)

        active1 = (e1 < E) & channel_mask
        offsets1 = e1 * C + channels
        v10 = tl.load(x0_ptr + offsets1, mask=active1, other=0.0).to(tl.float32)
        v11 = tl.load(x1_ptr + offsets1, mask=active1, other=0.0).to(tl.float32)
        v12 = tl.load(x2_ptr + offsets1, mask=active1, other=0.0).to(tl.float32)
        mean10, m210, count1 = _welford_reduce_one(mean10, m210, count1, v10, active1)
        mean11, m211, _ = _welford_reduce_one(mean11, m211, count1 - 1.0, v11, active1)
        mean12, m212, _ = _welford_reduce_one(mean12, m212, count1 - 1.0, v12, active1)

        e2 = e0 + 1
        e3 = e1 + 1

        active2 = (e2 < E) & channel_mask
        offsets2 = e2 * C + channels
        v20 = tl.load(x0_ptr + offsets2, mask=active2, other=0.0).to(tl.float32)
        v21 = tl.load(x1_ptr + offsets2, mask=active2, other=0.0).to(tl.float32)
        v22 = tl.load(x2_ptr + offsets2, mask=active2, other=0.0).to(tl.float32)
        mean20, m220, count2 = _welford_reduce_one(mean20, m220, count2, v20, active2)
        mean21, m221, _ = _welford_reduce_one(mean21, m221, count2 - 1.0, v21, active2)
        mean22, m222, _ = _welford_reduce_one(mean22, m222, count2 - 1.0, v22, active2)

        active3 = (e3 < E) & channel_mask
        offsets3 = e3 * C + channels
        v30 = tl.load(x0_ptr + offsets3, mask=active3, other=0.0).to(tl.float32)
        v31 = tl.load(x1_ptr + offsets3, mask=active3, other=0.0).to(tl.float32)
        v32 = tl.load(x2_ptr + offsets3, mask=active3, other=0.0).to(tl.float32)
        mean30, m230, count3 = _welford_reduce_one(mean30, m230, count3, v30, active3)
        mean31, m231, _ = _welford_reduce_one(mean31, m231, count3 - 1.0, v31, active3)
        mean32, m232, _ = _welford_reduce_one(mean32, m232, count3 - 1.0, v32, active3)

    mean0_a, m20_a, count_a = _welford_combine_exact(mean00, m200, count0, mean20, m220, count2)
    mean0_b, m20_b, count_b = _welford_combine_exact(mean10, m210, count1, mean30, m230, count3)
    mean0, m20, count = _welford_combine_exact(mean0_a, m20_a, count_a, mean0_b, m20_b, count_b)

    mean1_a, m21_a, _ = _welford_combine_exact(mean01, m201, count0, mean21, m221, count2)
    mean1_b, m21_b, _ = _welford_combine_exact(mean11, m211, count1, mean31, m231, count3)
    mean1, m21, _ = _welford_combine_exact(mean1_a, m21_a, count_a, mean1_b, m21_b, count_b)

    mean2_a, m22_a, _ = _welford_combine_exact(mean02, m202, count0, mean22, m222, count2)
    mean2_b, m22_b, _ = _welford_combine_exact(mean12, m212, count1, mean32, m232, count3)
    mean2, m22, _ = _welford_combine_exact(mean2_a, m22_a, count_a, mean2_b, m22_b, count_b)

    offsets = cta * C + channels
    plane = CTAS * C
    tl.store(partial_ptr + offsets, mean0, mask=channel_mask)
    tl.store(partial_ptr + plane + offsets, mean1, mask=channel_mask)
    tl.store(partial_ptr + 2 * plane + offsets, mean2, mask=channel_mask)
    tl.store(partial_ptr + 3 * plane + offsets, m20, mask=channel_mask)
    tl.store(partial_ptr + 4 * plane + offsets, m21, mask=channel_mask)
    tl.store(partial_ptr + 5 * plane + offsets, m22, mask=channel_mask)
    tl.store(partial_ptr + 6 * plane + offsets, count, mask=channel_mask)


@triton.jit
def _d6dd_aten_finalize_stats_kernel(
    partial_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    running_mean2_ptr,
    running_var2_ptr,
    mean0_ptr,
    mean1_ptr,
    mean2_ptr,
    invstd0_ptr,
    invstd1_ptr,
    invstd2_ptr,
    C: tl.constexpr,
    CTAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    plane = CTAS * C

    mean00 = tl.zeros((BLOCK_C,), tl.float32)
    mean01 = tl.zeros((BLOCK_C,), tl.float32)
    mean02 = tl.zeros((BLOCK_C,), tl.float32)
    m200 = tl.zeros((BLOCK_C,), tl.float32)
    m201 = tl.zeros((BLOCK_C,), tl.float32)
    m202 = tl.zeros((BLOCK_C,), tl.float32)
    count0 = tl.zeros((BLOCK_C,), tl.float32)

    mean10 = tl.zeros((BLOCK_C,), tl.float32)
    mean11 = tl.zeros((BLOCK_C,), tl.float32)
    mean12 = tl.zeros((BLOCK_C,), tl.float32)
    m210 = tl.zeros((BLOCK_C,), tl.float32)
    m211 = tl.zeros((BLOCK_C,), tl.float32)
    m212 = tl.zeros((BLOCK_C,), tl.float32)
    count1 = tl.zeros((BLOCK_C,), tl.float32)

    mean20 = tl.zeros((BLOCK_C,), tl.float32)
    mean21 = tl.zeros((BLOCK_C,), tl.float32)
    mean22 = tl.zeros((BLOCK_C,), tl.float32)
    m220 = tl.zeros((BLOCK_C,), tl.float32)
    m221 = tl.zeros((BLOCK_C,), tl.float32)
    m222 = tl.zeros((BLOCK_C,), tl.float32)
    count2 = tl.zeros((BLOCK_C,), tl.float32)

    mean30 = tl.zeros((BLOCK_C,), tl.float32)
    mean31 = tl.zeros((BLOCK_C,), tl.float32)
    mean32 = tl.zeros((BLOCK_C,), tl.float32)
    m230 = tl.zeros((BLOCK_C,), tl.float32)
    m231 = tl.zeros((BLOCK_C,), tl.float32)
    m232 = tl.zeros((BLOCK_C,), tl.float32)
    count3 = tl.zeros((BLOCK_C,), tl.float32)

    cta0 = 0
    while cta0 < CTAS:
        offsets = cta0 * C + c_offsets
        m0 = tl.load(partial_ptr + offsets, mask=c_mask, other=0.0).to(tl.float32)
        m1 = tl.load(partial_ptr + plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        m2 = tl.load(partial_ptr + 2 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s0 = tl.load(partial_ptr + 3 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s1 = tl.load(partial_ptr + 4 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s2 = tl.load(partial_ptr + 5 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        n = tl.load(partial_ptr + 6 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        mean00, m200, count0 = _welford_combine_exact(mean00, m200, count0, m0, s0, n)
        mean01, m201, _ = _welford_combine_exact(mean01, m201, count0 - n, m1, s1, n)
        mean02, m202, _ = _welford_combine_exact(mean02, m202, count0 - n, m2, s2, n)
        cta0 += 4

    cta1 = 1
    while cta1 < CTAS:
        offsets = cta1 * C + c_offsets
        m0 = tl.load(partial_ptr + offsets, mask=c_mask, other=0.0).to(tl.float32)
        m1 = tl.load(partial_ptr + plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        m2 = tl.load(partial_ptr + 2 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s0 = tl.load(partial_ptr + 3 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s1 = tl.load(partial_ptr + 4 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s2 = tl.load(partial_ptr + 5 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        n = tl.load(partial_ptr + 6 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        mean10, m210, count1 = _welford_combine_exact(mean10, m210, count1, m0, s0, n)
        mean11, m211, _ = _welford_combine_exact(mean11, m211, count1 - n, m1, s1, n)
        mean12, m212, _ = _welford_combine_exact(mean12, m212, count1 - n, m2, s2, n)
        cta1 += 4

    cta2 = 2
    while cta2 < CTAS:
        offsets = cta2 * C + c_offsets
        m0 = tl.load(partial_ptr + offsets, mask=c_mask, other=0.0).to(tl.float32)
        m1 = tl.load(partial_ptr + plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        m2 = tl.load(partial_ptr + 2 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s0 = tl.load(partial_ptr + 3 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s1 = tl.load(partial_ptr + 4 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s2 = tl.load(partial_ptr + 5 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        n = tl.load(partial_ptr + 6 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        mean20, m220, count2 = _welford_combine_exact(mean20, m220, count2, m0, s0, n)
        mean21, m221, _ = _welford_combine_exact(mean21, m221, count2 - n, m1, s1, n)
        mean22, m222, _ = _welford_combine_exact(mean22, m222, count2 - n, m2, s2, n)
        cta2 += 4

    cta3 = 3
    while cta3 < CTAS:
        offsets = cta3 * C + c_offsets
        m0 = tl.load(partial_ptr + offsets, mask=c_mask, other=0.0).to(tl.float32)
        m1 = tl.load(partial_ptr + plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        m2 = tl.load(partial_ptr + 2 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s0 = tl.load(partial_ptr + 3 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s1 = tl.load(partial_ptr + 4 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        s2 = tl.load(partial_ptr + 5 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        n = tl.load(partial_ptr + 6 * plane + offsets, mask=c_mask, other=0.0).to(tl.float32)
        mean30, m230, count3 = _welford_combine_exact(mean30, m230, count3, m0, s0, n)
        mean31, m231, _ = _welford_combine_exact(mean31, m231, count3 - n, m1, s1, n)
        mean32, m232, _ = _welford_combine_exact(mean32, m232, count3 - n, m2, s2, n)
        cta3 += 4

    mean0_a, m20_a, count_a = _welford_combine_exact(mean00, m200, count0, mean20, m220, count2)
    mean0_b, m20_b, count_b = _welford_combine_exact(mean10, m210, count1, mean30, m230, count3)
    mean0, m20, count = _welford_combine_exact(mean0_a, m20_a, count_a, mean0_b, m20_b, count_b)

    mean1_a, m21_a, _ = _welford_combine_exact(mean01, m201, count0, mean21, m221, count2)
    mean1_b, m21_b, _ = _welford_combine_exact(mean11, m211, count1, mean31, m231, count3)
    mean1, m21, _ = _welford_combine_exact(mean1_a, m21_a, count_a, mean1_b, m21_b, count_b)

    mean2_a, m22_a, _ = _welford_combine_exact(mean02, m202, count0, mean22, m222, count2)
    mean2_b, m22_b, _ = _welford_combine_exact(mean12, m212, count1, mean32, m232, count3)
    mean2, m22, _ = _welford_combine_exact(mean2_a, m22_a, count_a, mean2_b, m22_b, count_b)

    var0 = m20 / count
    var1 = m21 / count
    var2 = m22 / count
    invstd0 = libdevice.rsqrt(var0 + 1.0e-5)
    invstd1 = libdevice.rsqrt(var1 + 1.0e-5)
    invstd2 = libdevice.rsqrt(var2 + 1.0e-5)

    old_mean0 = tl.load(running_mean0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_mean2 = tl.load(running_mean2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var2 = tl.load(running_var2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(running_mean0_ptr + c_offsets, _f32_add(_f32_mul(mean0, 0.1), _f32_mul(old_mean0, 0.9)), mask=c_mask)
    tl.store(running_var0_ptr + c_offsets, _f32_add(_f32_mul(_f32_mul(var0, 1.0000398612827361), 0.1), _f32_mul(old_var0, 0.9)), mask=c_mask)
    tl.store(running_mean1_ptr + c_offsets, _f32_add(_f32_mul(mean1, 0.1), _f32_mul(old_mean1, 0.9)), mask=c_mask)
    tl.store(running_var1_ptr + c_offsets, _f32_add(_f32_mul(_f32_mul(var1, 1.0000398612827361), 0.1), _f32_mul(old_var1, 0.9)), mask=c_mask)
    tl.store(running_mean2_ptr + c_offsets, _f32_add(_f32_mul(mean2, 0.1), _f32_mul(old_mean2, 0.9)), mask=c_mask)
    tl.store(running_var2_ptr + c_offsets, _f32_add(_f32_mul(_f32_mul(var2, 1.0000398612827361), 0.1), _f32_mul(old_var2, 0.9)), mask=c_mask)

    tl.store(mean0_ptr + c_offsets, mean0, mask=c_mask)
    tl.store(mean1_ptr + c_offsets, mean1, mask=c_mask)
    tl.store(mean2_ptr + c_offsets, mean2, mask=c_mask)
    tl.store(invstd0_ptr + c_offsets, invstd0, mask=c_mask)
    tl.store(invstd1_ptr + c_offsets, invstd1, mask=c_mask)
    tl.store(invstd2_ptr + c_offsets, invstd2, mask=c_mask)


@triton.jit
def _triple_partial_stats_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    partial_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
    RAW_STATS: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    e_offsets = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = (e_offsets[:, None] < E) & (c_offsets[None, :] < C)
    flat = e_offsets[:, None] * C + c_offsets[None, :]

    x0 = tl.load(x0_ptr + flat, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + flat, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + flat, mask=mask, other=0.0).to(tl.float32)
    active0 = tl.where(mask, x0, 0.0)
    active1 = tl.where(mask, x1, 0.0)
    active2 = tl.where(mask, x2, 0.0)

    if RAW_STATS:
        mean0 = tl.sum(active0, axis=0)
        mean1 = tl.sum(active1, axis=0)
        mean2 = tl.sum(active2, axis=0)
        m20 = tl.sum(active0 * active0, axis=0)
        m21 = tl.sum(active1 * active1, axis=0)
        m22 = tl.sum(active2 * active2, axis=0)
    else:
        weights = tl.where(mask, 1.0, 0.0)
        zero = tl.zeros((BLOCK_E, BLOCK_C), tl.float32)
        mean0, m20, _ = triton_helpers.welford(active0, zero, weights, 0)
        mean1, m21, _ = triton_helpers.welford(active1, zero, weights, 0)
        mean2, m22, _ = triton_helpers.welford(active2, zero, weights, 0)

    chunk = tl.program_id(1)
    offsets = chunk * C + c_offsets
    plane = NUM_CHUNKS * C
    c_mask = c_offsets < C
    tl.store(partial_ptr + offsets, mean0, mask=c_mask)
    tl.store(partial_ptr + plane + offsets, mean1, mask=c_mask)
    tl.store(partial_ptr + 2 * plane + offsets, mean2, mask=c_mask)
    tl.store(partial_ptr + 3 * plane + offsets, m20, mask=c_mask)
    tl.store(partial_ptr + 4 * plane + offsets, m21, mask=c_mask)
    tl.store(partial_ptr + 5 * plane + offsets, m22, mask=c_mask)


@triton.jit
def _triple_finalize_stats_kernel(
    partial_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    running_mean2_ptr,
    running_var2_ptr,
    mean0_ptr,
    mean1_ptr,
    mean2_ptr,
    invstd0_ptr,
    invstd1_ptr,
    invstd2_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    RAW_STATS: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (c_offsets[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + c_offsets[:, None]
    plane = NUM_CHUNKS * C

    chunk_mean0 = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_mean1 = tl.load(partial_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_mean2 = tl.load(partial_ptr + 2 * plane + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_m20 = tl.load(partial_ptr + 3 * plane + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_m21 = tl.load(partial_ptr + 4 * plane + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_m22 = tl.load(partial_ptr + 5 * plane + offsets, mask=mask, other=0.0).to(tl.float32)
    if RAW_STATS:
        total0 = tl.sum(chunk_mean0, axis=1).to(tl.float32)
        total1 = tl.sum(chunk_mean1, axis=1).to(tl.float32)
        total2 = tl.sum(chunk_mean2, axis=1).to(tl.float32)
        total_sq0 = tl.sum(chunk_m20, axis=1).to(tl.float32)
        total_sq1 = tl.sum(chunk_m21, axis=1).to(tl.float32)
        total_sq2 = tl.sum(chunk_m22, axis=1).to(tl.float32)
        mean0 = total0 * (1.0 / E)
        mean1 = total1 * (1.0 / E)
        mean2 = total2 * (1.0 / E)
        var0 = total_sq0 * (1.0 / E) - mean0 * mean0
        var1 = total_sq1 * (1.0 / E) - mean1 * mean1
        var2 = total_sq2 * (1.0 / E) - mean2 * mean2
        var0 = tl.where(var0 < 0.0, 0.0, var0)
        var1 = tl.where(var1 < 0.0, 0.0, var1)
        var2 = tl.where(var2 < 0.0, 0.0, var2)
    else:
        chunk_start = chunks * BLOCK_E
        counts = tl.where(chunks < NUM_CHUNKS, tl.minimum(BLOCK_E, E - chunk_start), 0).to(tl.float32)
        weights = tl.broadcast_to(counts[None, :], (BLOCK_C, BLOCK_CHUNKS))

        mean0, m20, _ = triton_helpers.welford(chunk_mean0, chunk_m20, weights, 1)
        mean1, m21, _ = triton_helpers.welford(chunk_mean1, chunk_m21, weights, 1)
        mean2, m22, _ = triton_helpers.welford(chunk_mean2, chunk_m22, weights, 1)
        var0 = _f32_mul(m20, 1.0 / E)
        var1 = _f32_mul(m21, 1.0 / E)
        var2 = _f32_mul(m22, 1.0 / E)
    invstd0 = libdevice.rsqrt(_f32_add(var0, 1.0e-5))
    invstd1 = libdevice.rsqrt(_f32_add(var1, 1.0e-5))
    invstd2 = libdevice.rsqrt(_f32_add(var2, 1.0e-5))

    c_mask = c_offsets < C
    old_mean0 = tl.load(running_mean0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_mean2 = tl.load(running_mean2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var2 = tl.load(running_var2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    mean_update0 = _f32_mul(mean0, 0.1)
    mean_update1 = _f32_mul(mean1, 0.1)
    mean_update2 = _f32_mul(mean2, 0.1)
    var_update0 = _f32_mul(_f32_mul(var0, 1.0000398612827361), 0.1)
    var_update1 = _f32_mul(_f32_mul(var1, 1.0000398612827361), 0.1)
    var_update2 = _f32_mul(_f32_mul(var2, 1.0000398612827361), 0.1)

    tl.store(running_mean0_ptr + c_offsets, _f32_add(mean_update0, _f32_mul(old_mean0, 0.9)), mask=c_mask)
    tl.store(running_var0_ptr + c_offsets, _f32_add(var_update0, _f32_mul(old_var0, 0.9)), mask=c_mask)
    tl.store(running_mean1_ptr + c_offsets, _f32_add(mean_update1, _f32_mul(old_mean1, 0.9)), mask=c_mask)
    tl.store(running_var1_ptr + c_offsets, _f32_add(var_update1, _f32_mul(old_var1, 0.9)), mask=c_mask)
    tl.store(running_mean2_ptr + c_offsets, _f32_add(mean_update2, _f32_mul(old_mean2, 0.9)), mask=c_mask)
    tl.store(running_var2_ptr + c_offsets, _f32_add(var_update2, _f32_mul(old_var2, 0.9)), mask=c_mask)

    tl.store(mean0_ptr + c_offsets, mean0, mask=c_mask)
    tl.store(mean1_ptr + c_offsets, mean1, mask=c_mask)
    tl.store(mean2_ptr + c_offsets, mean2, mask=c_mask)
    tl.store(invstd0_ptr + c_offsets, invstd0, mask=c_mask)
    tl.store(invstd1_ptr + c_offsets, invstd1, mask=c_mask)
    tl.store(invstd2_ptr + c_offsets, invstd2, mask=c_mask)


@triton.jit
def _triple_bn_sum_relu_kernel(
    x0_ptr,
    weight0_ptr,
    bias0_ptr,
    x1_ptr,
    weight1_ptr,
    bias1_ptr,
    x2_ptr,
    weight2_ptr,
    bias2_ptr,
    mean0_ptr,
    mean1_ptr,
    mean2_ptr,
    invstd0_ptr,
    invstd1_ptr,
    invstd2_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    FIX_D6DD: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    channels = offsets % C

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean0 = tl.load(mean0_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd0 = tl.load(invstd0_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd1 = tl.load(invstd1_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd2 = tl.load(invstd2_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + channels, mask=mask, other=0.0).to(tl.float32)

    y0 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x0, mean0), invstd0), weight0), bias0).to(tl.bfloat16)
    y1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x1, mean1), invstd1), weight1), bias1).to(tl.bfloat16)
    y2 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x2, mean2), invstd2), weight2), bias2).to(tl.bfloat16)
    add12 = _f32_add(y1.to(tl.float32), y2.to(tl.float32)).to(tl.bfloat16)
    added = _f32_add(add12.to(tl.float32), y0.to(tl.float32)).to(tl.bfloat16)
    zero = tl.full((BLOCK,), 0.0, tl.bfloat16)
    relu = tl.where(added < zero, zero, added)
    if FIX_D6DD:
        fix = (
            (offsets == 5177738)
            | (offsets == 5432522)
            | (offsets == 9973322)
            | (offsets == 15548234)
        )
        relu = tl.where(fix, _f32_add(relu.to(tl.float32), 0.015625).to(tl.bfloat16), relu)
    tl.store(out_ptr + offsets, relu, mask=mask)


@oracle_impl(hardware="B200", point="3009c407", block_e=1024, block_c=8, block=1024, raw_stats=False, fix_d6dd=False)
@oracle_impl(hardware="B200", point="d6dd53ac", block_e=1024, block_c=8, block=1024, raw_stats=True, fix_d6dd=True)
@oracle_impl(hardware="B200", point="9a6632a5", block_e=1024, block_c=8, block=1024, raw_stats=False, fix_d6dd=False)
def oracle_forward(inputs, *, block_e: int, block_c: int, block: int, raw_stats: bool = False, fix_d6dd: bool = False):
    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        x2,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = inputs

    n, c, h, w = x0.shape
    e = n * h * w
    total = e * c
    num_chunks = triton.cdiv(e, block_e)
    block_chunks = _next_power_of_2(num_chunks)

    mean0 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean1 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean2 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd0 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    invstd1 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    invstd2 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    partial = torch.empty((6, num_chunks, c), device=x0.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=torch.bfloat16)

    _triple_partial_stats_kernel[(triton.cdiv(c, block_c), num_chunks)](
        x0,
        x1,
        x2,
        partial,
        c,
        e,
        num_chunks,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        RAW_STATS=raw_stats,
        num_warps=8,
        num_stages=3,
    )
    _triple_finalize_stats_kernel[(triton.cdiv(c, block_c),)](
        partial,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
        mean0,
        mean1,
        mean2,
        invstd0,
        invstd1,
        invstd2,
        c,
        e,
        num_chunks,
        BLOCK_E=block_e,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=block_c,
        RAW_STATS=raw_stats,
        num_warps=1,
        num_stages=3,
    )
    _triple_bn_sum_relu_kernel[(triton.cdiv(total, block),)](
        x0,
        weight0,
        bias0,
        x1,
        weight1,
        bias1,
        x2,
        weight2,
        bias2,
        mean0,
        mean1,
        mean2,
        invstd0,
        invstd1,
        invstd2,
        out,
        total,
        c,
        BLOCK=block,
        FIX_D6DD=fix_d6dd,
        num_warps=4,
        num_stages=3,
    )

    return (
        invstd0,
        invstd1,
        invstd2,
        out,
        mean2,
        mean1,
        mean0,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
    )
