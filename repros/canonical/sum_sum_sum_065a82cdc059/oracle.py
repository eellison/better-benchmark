"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full bf16 RepVGG chained BatchNorm-backward fragment with a first split-K channel reduction, an explicitly rounded residual/where producer, one shared downstream three-accumulator split reduction for the duplicated producer sum and two centered-dot sums, and the two dependent dense bf16 epilogues plus all returned fp32 vectors; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K template that carries finalized per-channel state through a rounded bf16 producer and co-finalizes sibling reductions feeding multiple tensor/vector epilogues, so it schedules the upstream reduction, rounded producer, duplicate downstream channel reductions, and broadcast epilogues as generic kernels; the fix is COOPERATIVE_SPLIT_K: add a stride-guarded multi-output channel-reduction template that preserves dtype boundaries while sharing compatible reductions and dependent epilogues across the full repro scope."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


INV_NHW = 2.4912308673469386e-06


def _ceil_pow2(value: int) -> int:
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
def _stage1_partial_kernel(
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    partial_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    A2_SN: tl.constexpr,
    A2_SC: tl.constexpr,
    A2_SH: tl.constexpr,
    A2_SW: tl.constexpr,
    A3_SN: tl.constexpr,
    A3_SC: tl.constexpr,
    A3_SH: tl.constexpr,
    A3_SW: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    active = (c_offsets < C) & (k_offsets < K)

    n = k_offsets // (H * W)
    spatial = k_offsets - n * (H * W)
    h = spatial // W
    w = spatial - h * W

    off2 = n * A2_SN + c_offsets * A2_SC + h * A2_SH + w * A2_SW
    off3 = n * A3_SN + c_offsets * A3_SC + h * A3_SH + w * A3_SW

    grad = tl.load(arg2_ptr + off2, mask=active, other=0.0).to(tl.float32)
    src = tl.load(arg3_ptr + off3, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg4_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered = _f32_sub(src, mean)

    grad_active = tl.where(active, grad, 0.0)
    prod = tl.where(active, _f32_mul(grad, centered), 0.0)

    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    partial_offsets = tl.program_id(1) * C + c_vec
    plane_stride = NUM_K_BLOCKS * C
    tl.store(partial_ptr + partial_offsets, tl.sum(grad_active, axis=0), mask=c_vec < C)
    tl.store(
        partial_ptr + plane_stride + partial_offsets,
        tl.sum(prod, axis=0),
        mask=c_vec < C,
    )


@triton.jit
def _stage1_finalize_kernel(
    partial_ptr,
    arg5_ptr,
    arg6_ptr,
    stats_ptr,
    sum_out_ptr,
    vec_out_ptr,
    C: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INV_NHW_: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    p_offsets = tl.arange(0, BLOCK_P)[:, None]
    active = (c_offsets < C) & (p_offsets < NUM_K_BLOCKS)
    offsets = p_offsets * C + c_offsets
    plane_stride = NUM_K_BLOCKS * C

    sum_grad = tl.sum(
        tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    sum_prod = tl.sum(
        tl.load(partial_ptr + plane_stride + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )

    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    channel = c_vec < C
    invstd = tl.load(arg5_ptr + c_vec, mask=channel, other=0.0).to(tl.float32)
    weight = tl.load(arg6_ptr + c_vec, mask=channel, other=0.0).to(tl.float32)

    mean_grad = _f32_mul(sum_grad, INV_NHW_)
    prod_mean = _f32_mul(sum_prod, INV_NHW_)
    invstd_sq = _f32_mul(invstd, invstd)
    coeff = _f32_mul(prod_mean, invstd_sq)
    scale = _f32_mul(invstd, weight)

    tl.store(stats_ptr + c_vec, mean_grad, mask=channel)
    tl.store(stats_ptr + C + c_vec, coeff, mask=channel)
    tl.store(stats_ptr + C * 2 + c_vec, scale, mask=channel)
    tl.store(sum_out_ptr + c_vec, sum_grad, mask=channel)
    tl.store(vec_out_ptr + c_vec, _f32_mul(sum_prod, invstd), mask=channel)


@triton.jit
def _producer_downstream_partial_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg7_ptr,
    arg8_ptr,
    arg9_ptr,
    arg12_ptr,
    arg13_ptr,
    stats1_ptr,
    producer_work_ptr,
    partial_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    A0_SN: tl.constexpr,
    A0_SC: tl.constexpr,
    A0_SH: tl.constexpr,
    A0_SW: tl.constexpr,
    A1_SN: tl.constexpr,
    A1_SC: tl.constexpr,
    A1_SH: tl.constexpr,
    A1_SW: tl.constexpr,
    A2_SN: tl.constexpr,
    A2_SC: tl.constexpr,
    A2_SH: tl.constexpr,
    A2_SW: tl.constexpr,
    A3_SN: tl.constexpr,
    A3_SC: tl.constexpr,
    A3_SH: tl.constexpr,
    A3_SW: tl.constexpr,
    A8_SN: tl.constexpr,
    A8_SC: tl.constexpr,
    A8_SH: tl.constexpr,
    A8_SW: tl.constexpr,
    A12_SN: tl.constexpr,
    A12_SC: tl.constexpr,
    A12_SH: tl.constexpr,
    A12_SW: tl.constexpr,
    WORK_SN: tl.constexpr,
    WORK_SC: tl.constexpr,
    WORK_SH: tl.constexpr,
    WORK_SW: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    active = (c_offsets < C) & (k_offsets < K)

    n = k_offsets // (H * W)
    spatial = k_offsets - n * (H * W)
    h = spatial // W
    w = spatial - h * W

    off0 = n * A0_SN + c_offsets * A0_SC + h * A0_SH + w * A0_SW
    off1 = n * A1_SN + c_offsets * A1_SC + h * A1_SH + w * A1_SW
    off2 = n * A2_SN + c_offsets * A2_SC + h * A2_SH + w * A2_SW
    off3 = n * A3_SN + c_offsets * A3_SC + h * A3_SH + w * A3_SW
    off8 = n * A8_SN + c_offsets * A8_SC + h * A8_SH + w * A8_SW
    off12 = n * A12_SN + c_offsets * A12_SC + h * A12_SH + w * A12_SW
    work_off = n * WORK_SN + c_offsets * WORK_SC + h * WORK_SH + w * WORK_SW

    lhs = tl.load(arg0_ptr + off0, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + off1, mask=active, other=0.0).to(tl.float32)
    residual = _f32_add(lhs, rhs).to(tl.bfloat16).to(tl.float32)

    grad = tl.load(arg2_ptr + off2, mask=active, other=0.0).to(tl.float32)
    gate = tl.load(arg3_ptr + off3, mask=active, other=0.0).to(tl.float32)
    mean1 = tl.load(arg4_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered1 = _f32_sub(gate, mean1)

    grad_mean = tl.load(stats1_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    coeff1 = tl.load(stats1_ptr + C + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    scale1 = tl.load(stats1_ptr + C * 2 + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)

    dense1 = _f32_sub(grad, _f32_mul(centered1, coeff1))
    dense1 = _f32_sub(dense1, grad_mean)
    dense1 = _f32_mul(dense1, scale1).to(tl.bfloat16).to(tl.float32)
    combined = _f32_add(residual, dense1).to(tl.bfloat16).to(tl.float32)

    fill = tl.load(arg7_ptr).to(tl.float32)
    producer = tl.where(gate <= 0.0, fill, combined).to(tl.bfloat16).to(tl.float32)
    tl.store(producer_work_ptr + work_off, producer.to(tl.bfloat16), mask=active)

    mean2 = tl.load(arg9_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    mean3 = tl.load(arg13_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered2 = _f32_sub(
        tl.load(arg8_ptr + off8, mask=active, other=0.0).to(tl.float32),
        mean2,
    )
    centered3 = _f32_sub(
        tl.load(arg12_ptr + off12, mask=active, other=0.0).to(tl.float32),
        mean3,
    )

    producer_active = tl.where(active, producer, 0.0)
    dot2 = tl.where(active, _f32_mul(producer, centered2), 0.0)
    dot3 = tl.where(active, _f32_mul(producer, centered3), 0.0)

    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    partial_offsets = tl.program_id(1) * C + c_vec
    plane_stride = NUM_K_BLOCKS * C
    tl.store(partial_ptr + partial_offsets, tl.sum(producer_active, axis=0), mask=c_vec < C)
    tl.store(
        partial_ptr + plane_stride + partial_offsets,
        tl.sum(dot2, axis=0),
        mask=c_vec < C,
    )
    tl.store(
        partial_ptr + plane_stride * 2 + partial_offsets,
        tl.sum(dot3, axis=0),
        mask=c_vec < C,
    )


@triton.jit
def _stage23_finalize_kernel(
    partial_ptr,
    arg10_ptr,
    arg11_ptr,
    arg14_ptr,
    arg15_ptr,
    stats_ptr,
    sum2_out_ptr,
    vec2_out_ptr,
    sum3_out_ptr,
    vec3_out_ptr,
    C: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INV_NHW_: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    p_offsets = tl.arange(0, BLOCK_P)[:, None]
    active = (c_offsets < C) & (p_offsets < NUM_K_BLOCKS)
    offsets = p_offsets * C + c_offsets
    plane_stride = NUM_K_BLOCKS * C

    sum_x = tl.sum(
        tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    sum_dot2 = tl.sum(
        tl.load(partial_ptr + plane_stride + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    sum_dot3 = tl.sum(
        tl.load(partial_ptr + plane_stride * 2 + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )

    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    channel = c_vec < C
    invstd2 = tl.load(arg10_ptr + c_vec, mask=channel, other=0.0).to(tl.float32)
    weight2 = tl.load(arg11_ptr + c_vec, mask=channel, other=0.0).to(tl.float32)
    invstd3 = tl.load(arg14_ptr + c_vec, mask=channel, other=0.0).to(tl.float32)
    weight3 = tl.load(arg15_ptr + c_vec, mask=channel, other=0.0).to(tl.float32)

    mean_x = _f32_mul(sum_x, INV_NHW_)
    dot2_mean = _f32_mul(sum_dot2, INV_NHW_)
    dot3_mean = _f32_mul(sum_dot3, INV_NHW_)
    coeff2 = _f32_mul(dot2_mean, _f32_mul(invstd2, invstd2))
    coeff3 = _f32_mul(dot3_mean, _f32_mul(invstd3, invstd3))
    scale2 = _f32_mul(invstd2, weight2)
    scale3 = _f32_mul(invstd3, weight3)

    tl.store(stats_ptr + c_vec, mean_x, mask=channel)
    tl.store(stats_ptr + C + c_vec, coeff2, mask=channel)
    tl.store(stats_ptr + C * 2 + c_vec, scale2, mask=channel)
    tl.store(stats_ptr + C * 3 + c_vec, coeff3, mask=channel)
    tl.store(stats_ptr + C * 4 + c_vec, scale3, mask=channel)
    tl.store(sum2_out_ptr + c_vec, sum_x, mask=channel)
    tl.store(sum3_out_ptr + c_vec, sum_x, mask=channel)
    tl.store(vec2_out_ptr + c_vec, _f32_mul(sum_dot2, invstd2), mask=channel)
    tl.store(vec3_out_ptr + c_vec, _f32_mul(sum_dot3, invstd3), mask=channel)


@triton.jit
def _stage23_epilogue_kernel(
    producer_work_ptr,
    arg8_ptr,
    arg9_ptr,
    arg12_ptr,
    arg13_ptr,
    stats_ptr,
    out4_ptr,
    out7_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    WORK_SN: tl.constexpr,
    WORK_SC: tl.constexpr,
    WORK_SH: tl.constexpr,
    WORK_SW: tl.constexpr,
    A8_SN: tl.constexpr,
    A8_SC: tl.constexpr,
    A8_SH: tl.constexpr,
    A8_SW: tl.constexpr,
    A12_SN: tl.constexpr,
    A12_SC: tl.constexpr,
    A12_SH: tl.constexpr,
    A12_SW: tl.constexpr,
    O4_SN: tl.constexpr,
    O4_SC: tl.constexpr,
    O4_SH: tl.constexpr,
    O4_SW: tl.constexpr,
    O7_SN: tl.constexpr,
    O7_SC: tl.constexpr,
    O7_SH: tl.constexpr,
    O7_SW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    active = (c_offsets < C) & (k_offsets < K)

    n = k_offsets // (H * W)
    spatial = k_offsets - n * (H * W)
    h = spatial // W
    w = spatial - h * W

    work_off = n * WORK_SN + c_offsets * WORK_SC + h * WORK_SH + w * WORK_SW
    off8 = n * A8_SN + c_offsets * A8_SC + h * A8_SH + w * A8_SW
    off12 = n * A12_SN + c_offsets * A12_SC + h * A12_SH + w * A12_SW
    out4_off = n * O4_SN + c_offsets * O4_SC + h * O4_SH + w * O4_SW
    out7_off = n * O7_SN + c_offsets * O7_SC + h * O7_SH + w * O7_SW

    producer = tl.load(producer_work_ptr + work_off, mask=active, other=0.0).to(tl.float32)
    mean2 = tl.load(arg9_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    mean3 = tl.load(arg13_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered2 = _f32_sub(
        tl.load(arg8_ptr + off8, mask=active, other=0.0).to(tl.float32),
        mean2,
    )
    centered3 = _f32_sub(
        tl.load(arg12_ptr + off12, mask=active, other=0.0).to(tl.float32),
        mean3,
    )

    mean_x = tl.load(stats_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    coeff2 = tl.load(stats_ptr + C + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    scale2 = tl.load(stats_ptr + C * 2 + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    coeff3 = tl.load(stats_ptr + C * 3 + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    scale3 = tl.load(stats_ptr + C * 4 + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)

    out_b = _f32_sub(producer, _f32_mul(centered2, coeff2))
    out_b = _f32_sub(out_b, mean_x)
    out_b = _f32_mul(out_b, scale2)
    out_c = _f32_sub(producer, _f32_mul(centered3, coeff3))
    out_c = _f32_sub(out_c, mean_x)
    out_c = _f32_mul(out_c, scale3)

    tl.store(out4_ptr + out4_off, out_b.to(tl.bfloat16), mask=active)
    tl.store(out7_ptr + out7_off, out_c.to(tl.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="f665655f", BLOCK_K=512, BLOCK_C=16, reduce_warps=8, final_warps=8, epilogue_warps=4)
@oracle_impl(hardware="B200", point="ff6d7f0c", BLOCK_K=256, BLOCK_C=16, reduce_warps=8, final_warps=8, epilogue_warps=4)
@oracle_impl(hardware="B200", point="c0fba172", BLOCK_K=256, BLOCK_C=16, reduce_warps=8, final_warps=4, epilogue_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
    ) = inputs
    n, c, h, w = (int(dim) for dim in arg2.shape)
    k = n * h * w
    num_k_blocks = triton.cdiv(k, BLOCK_K)
    block_p = _ceil_pow2(num_k_blocks)

    out4 = torch.empty_strided(
        tuple(arg8.shape),
        tuple(int(s) for s in arg8.stride()),
        device=arg8.device,
        dtype=torch.bfloat16,
    )
    out7 = torch.empty_strided(
        tuple(arg12.shape),
        tuple(int(s) for s in arg12.stride()),
        device=arg12.device,
        dtype=torch.bfloat16,
    )
    producer_work = torch.empty_strided(
        tuple(arg3.shape),
        tuple(int(s) for s in arg3.stride()),
        device=arg3.device,
        dtype=torch.bfloat16,
    )
    sum1 = torch.empty((c,), device=arg2.device, dtype=torch.float32)
    vec1 = torch.empty((c,), device=arg2.device, dtype=torch.float32)
    sum2 = torch.empty((c,), device=arg2.device, dtype=torch.float32)
    vec2 = torch.empty((c,), device=arg2.device, dtype=torch.float32)
    sum3 = torch.empty((c,), device=arg2.device, dtype=torch.float32)
    vec3 = torch.empty((c,), device=arg2.device, dtype=torch.float32)
    partial1 = torch.empty((2, num_k_blocks, c), device=arg2.device, dtype=torch.float32)
    partial23 = torch.empty((3, num_k_blocks, c), device=arg2.device, dtype=torch.float32)
    stats1 = torch.empty((3, c), device=arg2.device, dtype=torch.float32)
    stats23 = torch.empty((5, c), device=arg2.device, dtype=torch.float32)

    a0s = tuple(int(s) for s in arg0.stride())
    a1s = tuple(int(s) for s in arg1.stride())
    a2s = tuple(int(s) for s in arg2.stride())
    a3s = tuple(int(s) for s in arg3.stride())
    a8s = tuple(int(s) for s in arg8.stride())
    a12s = tuple(int(s) for s in arg12.stride())
    ws = tuple(int(s) for s in producer_work.stride())
    o4s = tuple(int(s) for s in out4.stride())
    o7s = tuple(int(s) for s in out7.stride())

    grid = (triton.cdiv(c, BLOCK_C), num_k_blocks)
    final_grid = (triton.cdiv(c, BLOCK_C),)
    _stage1_partial_kernel[grid](
        arg2,
        arg3,
        arg4,
        partial1,
        C=c,
        H=h,
        W=w,
        K=k,
        A2_SN=a2s[0],
        A2_SC=a2s[1],
        A2_SH=a2s[2],
        A2_SW=a2s[3],
        A3_SN=a3s[0],
        A3_SC=a3s[1],
        A3_SH=a3s[2],
        A3_SW=a3s[3],
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
    )
    _stage1_finalize_kernel[final_grid](
        partial1,
        arg5,
        arg6,
        stats1,
        sum1,
        vec1,
        C=c,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_P=block_p,
        BLOCK_C=BLOCK_C,
        INV_NHW_=INV_NHW,
        num_warps=final_warps,
    )
    _producer_downstream_partial_kernel[grid](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg7,
        arg8,
        arg9,
        arg12,
        arg13,
        stats1,
        producer_work,
        partial23,
        C=c,
        H=h,
        W=w,
        K=k,
        A0_SN=a0s[0],
        A0_SC=a0s[1],
        A0_SH=a0s[2],
        A0_SW=a0s[3],
        A1_SN=a1s[0],
        A1_SC=a1s[1],
        A1_SH=a1s[2],
        A1_SW=a1s[3],
        A2_SN=a2s[0],
        A2_SC=a2s[1],
        A2_SH=a2s[2],
        A2_SW=a2s[3],
        A3_SN=a3s[0],
        A3_SC=a3s[1],
        A3_SH=a3s[2],
        A3_SW=a3s[3],
        A8_SN=a8s[0],
        A8_SC=a8s[1],
        A8_SH=a8s[2],
        A8_SW=a8s[3],
        A12_SN=a12s[0],
        A12_SC=a12s[1],
        A12_SH=a12s[2],
        A12_SW=a12s[3],
        WORK_SN=ws[0],
        WORK_SC=ws[1],
        WORK_SH=ws[2],
        WORK_SW=ws[3],
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
    )
    _stage23_finalize_kernel[final_grid](
        partial23,
        arg10,
        arg11,
        arg14,
        arg15,
        stats23,
        sum2,
        vec2,
        sum3,
        vec3,
        C=c,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_P=block_p,
        BLOCK_C=BLOCK_C,
        INV_NHW_=INV_NHW,
        num_warps=final_warps,
    )
    _stage23_epilogue_kernel[grid](
        producer_work,
        arg8,
        arg9,
        arg12,
        arg13,
        stats23,
        out4,
        out7,
        C=c,
        H=h,
        W=w,
        K=k,
        WORK_SN=ws[0],
        WORK_SC=ws[1],
        WORK_SH=ws[2],
        WORK_SW=ws[3],
        A8_SN=a8s[0],
        A8_SC=a8s[1],
        A8_SH=a8s[2],
        A8_SW=a8s[3],
        A12_SN=a12s[0],
        A12_SC=a12s[1],
        A12_SH=a12s[2],
        A12_SW=a12s[3],
        O4_SN=o4s[0],
        O4_SC=o4s[1],
        O4_SH=o4s[2],
        O4_SW=o4s[3],
        O7_SN=o7s[0],
        O7_SC=o7s[1],
        O7_SH=o7s[2],
        O7_SW=o7s[3],
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=epilogue_warps,
    )
    return sum1, vec1, sum2, vec2, out4, sum3, vec3, out7
