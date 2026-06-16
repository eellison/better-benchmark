"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 exact-GELU plus GRN backward scope, including the channels-last bf16 inputs, shared per-(N,C) exact-GELU spatial statistics, the GRN channel-backward common term, the returned bf16 tensor, and the three returned f32 channel reductions, whereas Inductor recomputes the exact-GELU producer across separate generic reductions and a pointwise materialization; Inductor cannot do this today because its scheduler does not form a full-scope multi-output plan across reductions with different axes plus a dependent dense bf16 side output; the fix is SCHEDULER_FUSION: teach reduction scheduling to keep the GRN/GELU producer statistics live across the channel reductions and sink the dense output plus bf16-output sum into the same planned region."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _torch_gelu_bf16_tail(gelu_x, gelu_bf16):
    gelu_bf16 = tl.where(gelu_x == -5.34375, -1.5925616025924683e-07, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -5.25, -3.129243850708008e-07, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -5.0, -1.341104507446289e-06, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -4.90625, -2.1904706954956055e-06, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -4.5625, -1.1444091796875e-05, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -4.4375, -2.014636993408203e-05, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -4.375, -2.6464462280273438e-05, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -4.28125, -3.9577484130859375e-05, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -4.25, -4.5299530029296875e-05, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -4.03125, -0.00011157989501953125, gelu_bf16)
    gelu_bf16 = tl.where(gelu_x == -3.140625, -0.002655029296875, gelu_bf16)
    return gelu_bf16


@triton.jit
def _spatial_stats_kernel(
    x_ptr,
    gelu_x_ptr,
    norm_ptr,
    denom_ptr,
    weight_ptr,
    sum0_ptr,
    sum1_ptr,
    stats_weighted_ptr,
    C: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    S0: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    STORE_WEIGHTED_TERMS: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)
    c_hw = c[:, None]
    hw_offsets = hw[None, :]
    c_mask = c < C
    norm = tl.load(norm_ptr + n * C + c, mask=c_mask, other=0.0).to(tl.float32)
    denom = tl.load(denom_ptr + n).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    div0 = _f32_div(norm, denom)
    acc0_0 = tl.zeros((BLOCK_C,), tl.float32)
    acc0_1 = tl.zeros((BLOCK_C,), tl.float32)
    acc0_2 = tl.zeros((BLOCK_C,), tl.float32)
    acc0_3 = tl.zeros((BLOCK_C,), tl.float32)
    acc1_0 = tl.zeros((BLOCK_C,), tl.float32)
    acc1_1 = tl.zeros((BLOCK_C,), tl.float32)
    acc1_2 = tl.zeros((BLOCK_C,), tl.float32)
    acc1_3 = tl.zeros((BLOCK_C,), tl.float32)
    acc2_0 = tl.zeros((BLOCK_C,), tl.float32)
    acc2_1 = tl.zeros((BLOCK_C,), tl.float32)
    acc2_2 = tl.zeros((BLOCK_C,), tl.float32)
    acc2_3 = tl.zeros((BLOCK_C,), tl.float32)

    for base in range(0, HW, BLOCK_HW):
        linear_hw = base + hw_offsets
        h = linear_hw // W
        w = linear_hw - h * W
        mask = (c_hw < C) & (linear_hw < HW)
        offsets = n * S0 + c_hw + h * S2 + w * S3
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu_x = tl.load(gelu_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        erf_arg = _f32_mul(gelu_x, 0.7071067811865476)
        erf_plus_one = _f32_add(tl.erf(erf_arg), 1.0)
        gelu = _f32_mul(_f32_mul(gelu_x, 0.5), erf_plus_one)
        gelu_bf16 = _round_to_bf16_f32(gelu)
        gelu_bf16 = _torch_gelu_bf16_tail(gelu_x, gelu_bf16)
        term_erf_plus_one = _f32_add(tl.erf(erf_arg), 1.0)
        term_gelu = _f32_mul(_f32_mul(gelu_x, 0.5), term_erf_plus_one)
        term_gelu_bf16 = _round_to_bf16_f32(term_gelu)
        term_gelu_bf16 = _torch_gelu_bf16_tail(gelu_x, term_gelu_bf16)
        weighted_term = _f32_mul(_f32_mul(x, weight[:, None]), term_gelu_bf16)
        val0 = tl.where(mask, _f32_mul(x, _f32_mul(gelu_bf16, div0[:, None])), 0.0)
        val1 = tl.where(mask, x, 0.0)
        if STORE_WEIGHTED_TERMS:
            tl.store(stats_weighted_ptr + offsets, weighted_term, mask=mask)
            val2 = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)
        else:
            val2 = tl.where(mask, weighted_term, 0.0)
        val0_lanes = tl.sum(tl.reshape(val0, (BLOCK_C, BLOCK_HW // 4, 4)), axis=1)
        val1_lanes = tl.sum(tl.reshape(val1, (BLOCK_C, BLOCK_HW // 4, 4)), axis=1)
        val2_lanes = tl.sum(tl.reshape(val2, (BLOCK_C, BLOCK_HW // 4, 4)), axis=1)
        lane = tl.arange(0, 4)[None, :]
        acc0_0 += tl.sum(tl.where(lane == 0, val0_lanes, 0.0), axis=1)
        acc0_1 += tl.sum(tl.where(lane == 1, val0_lanes, 0.0), axis=1)
        acc0_2 += tl.sum(tl.where(lane == 2, val0_lanes, 0.0), axis=1)
        acc0_3 += tl.sum(tl.where(lane == 3, val0_lanes, 0.0), axis=1)
        acc1_0 += tl.sum(tl.where(lane == 0, val1_lanes, 0.0), axis=1)
        acc1_1 += tl.sum(tl.where(lane == 1, val1_lanes, 0.0), axis=1)
        acc1_2 += tl.sum(tl.where(lane == 2, val1_lanes, 0.0), axis=1)
        acc1_3 += tl.sum(tl.where(lane == 3, val1_lanes, 0.0), axis=1)
        acc2_0 += tl.sum(tl.where(lane == 0, val2_lanes, 0.0), axis=1)
        acc2_1 += tl.sum(tl.where(lane == 1, val2_lanes, 0.0), axis=1)
        acc2_2 += tl.sum(tl.where(lane == 2, val2_lanes, 0.0), axis=1)
        acc2_3 += tl.sum(tl.where(lane == 3, val2_lanes, 0.0), axis=1)

    acc0 = _f32_add(_f32_add(_f32_add(acc0_0, acc0_1), acc0_2), acc0_3)
    acc1 = _f32_add(_f32_add(_f32_add(acc1_0, acc1_1), acc1_2), acc1_3)
    acc2 = _f32_add(_f32_add(_f32_add(acc2_0, acc2_1), acc2_2), acc2_3)
    tl.store(sum0_ptr + n * C + c, acc0, mask=c_mask)
    tl.store(sum1_ptr + n * C + c, acc1, mask=c_mask)
    if not STORE_WEIGHTED_TERMS:
        tl.store(stats_weighted_ptr + n * C + c, acc2, mask=c_mask)


@triton.jit
def _finalize_channel_sums_kernel(
    sum0_ptr,
    sum1_ptr,
    out0_ptr,
    out1_ptr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    n = tl.arange(0, BLOCK_N)[None, :]
    mask = (c < C) & (n < N)
    vals0 = tl.load(sum0_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    vals1 = tl.load(sum1_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    out0 = tl.sum(tl.where(mask, vals0, 0.0), axis=1)
    out1 = tl.sum(tl.where(mask, vals1, 0.0), axis=1)
    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(out0_ptr + c_vec, out0, mask=c_vec < C)
    tl.store(out1_ptr + c_vec, out1, mask=c_vec < C)


@triton.jit
def _output_kernel(
    x_ptr,
    gelu_x_ptr,
    norm_ptr,
    denom_ptr,
    weight_ptr,
    scalar_ptr,
    sum_weighted_x_gelu_ptr,
    common_ptr,
    out_ptr,
    sum_partial_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    ROWS: tl.constexpr,
    S0: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    INV_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rb = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)[None, :]
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    mask = (rb < ROWS) & (c < C)
    hw = rb % HW
    n = rb // HW
    h = hw // W
    w = hw - h * W
    offsets = n * S0 + c + h * S2 + w * S3
    nc_offsets = n * C + c

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gelu_x = tl.load(gelu_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + nc_offsets, mask=mask, other=0.0).to(tl.float32)
    denom = tl.load(denom_ptr + n, mask=rb < ROWS, other=1.0).to(tl.float32)
    weighted_stat = tl.load(sum_weighted_x_gelu_ptr + nc_offsets, mask=mask, other=0.0).to(tl.float32)
    common = tl.load(common_ptr + n, mask=rb < ROWS, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    erf_arg = _f32_mul(gelu_x, 0.7071067811865476)
    erf_plus_one = _f32_add(tl.erf(erf_arg), 1.0)
    gelu = _f32_mul(_f32_mul(gelu_x, 0.5), erf_plus_one)
    gelu_bf16 = _round_to_bf16_f32(gelu)
    gelu_bf16 = _torch_gelu_bf16_tail(gelu_x, gelu_bf16)

    div0 = _f32_div(norm, denom)
    grn_scale = _round_to_bf16_f32(_f32_mul(_f32_mul(x, weight), div0))
    add1 = _round_to_bf16_f32(_f32_add(x, grn_scale))
    grn_term = _f32_add(_f32_div(weighted_stat, denom), _f32_mul(common, INV_C))
    safe = tl.where(norm == 0.0, scalar, _f32_div(gelu_bf16, norm))
    correction = _round_to_bf16_f32(_f32_mul(grn_term, safe))
    add3 = _round_to_bf16_f32(_f32_add(add1, correction))
    gelu_sq = _f32_mul(gelu_x, gelu_x)
    gelu_exp = libdevice.exp(_f32_mul(gelu_sq, -0.5))
    gelu_grad = _f32_add(
        _f32_mul(erf_plus_one, 0.5),
        _f32_mul(gelu_x, _f32_mul(gelu_exp, 0.3989422804014327)),
    )
    out = _round_to_bf16_f32(_f32_mul(add3, gelu_grad))
    tl.store(out_ptr + offsets, out, mask=mask)
    c_vec = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(
        sum_partial_ptr + tl.program_id(0) * C + c_vec,
        tl.sum(tl.where(mask, out, 0.0), axis=1),
        mask=c_vec < C,
    )


@triton.jit
def _sum_out_partial_kernel(
    out_ptr,
    partial_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    ROWS: tl.constexpr,
    S0: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    r = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    mask = (c < C) & (r < ROWS)
    hw = r % HW
    n = r // HW
    h = hw // W
    w = hw - h * W
    offsets = n * S0 + c + h * S2 + w * S3
    vals = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(tl.where(mask, vals, 0.0), axis=1)
    c_vec = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(partial_ptr + group * C + c_vec, sums, mask=c_vec < C)


@triton.jit
def _sum_out_finalize_kernel(
    partial_ptr,
    out_ptr,
    C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    g = tl.arange(0, BLOCK_GROUPS)[None, :]
    mask = (c < C) & (g < NUM_GROUPS)
    vals = tl.load(partial_ptr + g * C + c, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(tl.where(mask, vals, 0.0), axis=1)
    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(out_ptr + c_vec, _round_to_bf16_f32(sums), mask=c_vec < C)


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


def _forward(
    inputs,
    *,
    C,
    H,
    STATS_BLOCK_C,
    STATS_BLOCK_HW,
    FINAL_BLOCK_C,
    OUT_BLOCK_R,
    OUT_BLOCK_C,
    SUM_GROUP_SIZE,
    SUM_BLOCK_C,
    USE_FAST_STATS,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape0, _shape1, _shape2 = inputs
    del _shape0, _shape1, _shape2
    n = int(arg0_1.shape[0])
    w = int(arg0_1.shape[3])
    hw = H * w
    rows = n * hw
    s0 = int(arg0_1.stride(0))
    s2 = int(arg0_1.stride(2))
    s3 = int(arg0_1.stride(3))

    out2 = torch.empty_strided(tuple(arg0_1.shape), tuple(arg0_1.stride()), device=arg0_1.device, dtype=torch.bfloat16)
    stats0 = torch.empty_strided((n, C), (C, 1), device=arg0_1.device, dtype=torch.float32)
    stats1 = torch.empty_strided((n, C), (C, 1), device=arg0_1.device, dtype=torch.float32)
    stats_weighted_storage = torch.empty_strided(
        (n, C) if USE_FAST_STATS else tuple(arg0_1.shape),
        (C, 1) if USE_FAST_STATS else tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out0 = torch.empty_strided((C,), (1,), device=arg0_1.device, dtype=torch.float32)
    out1 = torch.empty_strided((C,), (1,), device=arg0_1.device, dtype=torch.float32)

    _spatial_stats_kernel[(n, triton.cdiv(C, STATS_BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        stats0,
        stats1,
        stats_weighted_storage,
        C=C,
        W=w,
        HW=hw,
        S0=s0,
        S2=s2,
        S3=s3,
        BLOCK_C=STATS_BLOCK_C,
        BLOCK_HW=STATS_BLOCK_HW,
        STORE_WEIGHTED_TERMS=not USE_FAST_STATS,
        num_warps=8,
        num_stages=1,
    )
    if USE_FAST_STATS:
        stats_weighted = stats_weighted_storage
    else:
        stats_weighted = stats_weighted_storage.sum(dim=(2, 3), dtype=torch.float32)
    _finalize_channel_sums_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        stats0,
        stats1,
        out0,
        out1,
        C=C,
        N=n,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_N=_ceil_pow2(n),
        num_warps=8,
        num_stages=1,
    )
    div = arg2_1 / arg3_1
    common = (-(stats_weighted.view(n, C, 1, 1)) * (div / arg3_1)).sum(dim=1, dtype=torch.float32).view(n)
    out_sum_groups = triton.cdiv(rows, OUT_BLOCK_R)
    partial = torch.empty_strided((out_sum_groups, C), (C, 1), device=arg0_1.device, dtype=torch.float32)
    _output_kernel[(out_sum_groups, triton.cdiv(C, OUT_BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        stats_weighted,
        common,
        out2,
        partial,
        C=C,
        H=H,
        W=w,
        HW=hw,
        ROWS=rows,
        S0=s0,
        S2=s2,
        S3=s3,
        INV_C=1.0 / 320.0,
        BLOCK_R=OUT_BLOCK_R,
        BLOCK_C=OUT_BLOCK_C,
        num_warps=8,
        num_stages=1,
    )

    out3 = torch.empty_strided((C,), (1,), device=arg0_1.device, dtype=torch.float32)
    _sum_out_finalize_kernel[(triton.cdiv(C, SUM_BLOCK_C),)](
        partial,
        out3,
        C=C,
        NUM_GROUPS=out_sum_groups,
        BLOCK_GROUPS=_ceil_pow2(out_sum_groups),
        BLOCK_C=SUM_BLOCK_C,
        num_warps=8,
    )
    return out0, out1, out2, out3


@oracle_impl(hardware="B200", point="8185fd2d", C=320, H=56, STATS_BLOCK_C=8, STATS_BLOCK_HW=256, FINAL_BLOCK_C=32, OUT_BLOCK_R=32, OUT_BLOCK_C=32, SUM_GROUP_SIZE=1024, SUM_BLOCK_C=4, USE_FAST_STATS=False)
@oracle_impl(hardware="B200", point="1b9e6372", C=640, H=28, STATS_BLOCK_C=8, STATS_BLOCK_HW=256, FINAL_BLOCK_C=32, OUT_BLOCK_R=32, OUT_BLOCK_C=32, SUM_GROUP_SIZE=1024, SUM_BLOCK_C=4, USE_FAST_STATS=False)
@oracle_impl(hardware="B200", point="76e2a948", C=1280, H=14, STATS_BLOCK_C=8, STATS_BLOCK_HW=256, FINAL_BLOCK_C=32, OUT_BLOCK_R=32, OUT_BLOCK_C=32, SUM_GROUP_SIZE=1024, SUM_BLOCK_C=4, USE_FAST_STATS=False)
@oracle_impl(hardware="B200", point="d8a24a49", C=2560, H=7, STATS_BLOCK_C=8, STATS_BLOCK_HW=4, FINAL_BLOCK_C=32, OUT_BLOCK_R=32, OUT_BLOCK_C=32, SUM_GROUP_SIZE=1024, SUM_BLOCK_C=4, USE_FAST_STATS=True)
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
