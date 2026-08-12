"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete RepVGG bf16 BN-backward-style scope with one shared masked producer, the visible channels-last f32 side tensor, shared channel reductions for `sum(where)`, `sum(where * centered1)`, and `sum(where * centered2)`, both returned sum side tensors, both returned scale-gradient vectors, and both channels-last bf16 dense gradient epilogues, whereas Inductor currently lowers the masked bf16 add/where producer, duplicate channel sums, two dependent BN-backward epilogues, bf16 casts, and visible side outputs as separate generic scheduler regions over large intermediates; Inductor cannot do this today because its scheduler/codegen does not build a full-scope multi-output reduction plan that keeps the shared masked producer and channel summaries reusable across sibling vector and dense epilogues while preserving bf16 rounding and returned-output layout; the fix is SCHEDULER_FUSION: add scheduler support for sibling channel reductions with shared finalized summaries and direct channels-last full-tensor/vector side-output stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


CAPTURED_INV_NHW = 2.4912308673469386e-06


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _partial_activation_sums_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    scalar_ptr,
    arg4_ptr,
    mean1_ptr,
    arg8_ptr,
    mean2_ptr,
    out0_ptr,
    partial_ptr,
    CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    HW: tl.constexpr,
    NHW: tl.constexpr,
    NUM_TILES: tl.constexpr,
    STRIDE_N: tl.constexpr,
    STRIDE_C: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    c_block = tl.program_id(1)
    m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    active = (m[:, None] < NHW) & (c[None, :] < CHANNELS)

    n = m // HW
    spatial = m - n * HW
    h = spatial // WIDTH
    w = spatial - h * WIDTH
    offsets = (
        n[:, None] * STRIDE_N
        + c[None, :] * STRIDE_C
        + h[:, None] * STRIDE_H
        + w[:, None] * STRIDE_W
    )

    lhs = tl.load(arg0_ptr + offsets, mask=active, other=0.0)
    rhs = tl.load(arg1_ptr + offsets, mask=active, other=0.0)
    selector = tl.load(arg2_ptr + offsets, mask=active, other=0.0)
    full_value = tl.load(scalar_ptr)
    added = _add_rn_f32(lhs.to(tl.float32), rhs.to(tl.float32)).to(tl.bfloat16)
    zero_bf16 = tl.full((BLOCK_M, BLOCK_C), 0.0, tl.float32).to(tl.bfloat16)
    where_bf16 = tl.where(selector <= zero_bf16, full_value, added).to(tl.bfloat16)
    where_f32 = where_bf16.to(tl.float32)
    tl.store(out0_ptr + offsets, where_f32, mask=active)

    mean1 = tl.load(mean1_ptr + c, mask=c < CHANNELS, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c, mask=c < CHANNELS, other=0.0).to(tl.float32)
    x1 = tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    x2 = tl.load(arg8_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    centered1 = _sub_rn_f32(x1, mean1[None, :])
    centered2 = _sub_rn_f32(x2, mean2[None, :])
    zero_f32 = tl.full((BLOCK_M, BLOCK_C), 0.0, tl.float32)
    reduced_where = tl.where(active, where_f32, zero_f32)
    dot1 = tl.where(active, _mul_rn_f32(where_f32, centered1), zero_f32)
    dot2 = tl.where(active, _mul_rn_f32(where_f32, centered2), zero_f32)

    partial_offsets = tile * CHANNELS + c
    plane = NUM_TILES * CHANNELS
    c_mask = c < CHANNELS
    tl.store(partial_ptr + partial_offsets, tl.sum(reduced_where, axis=0), mask=c_mask)
    tl.store(partial_ptr + plane + partial_offsets, tl.sum(dot1, axis=0), mask=c_mask)
    tl.store(partial_ptr + plane * 2 + partial_offsets, tl.sum(dot2, axis=0), mask=c_mask)


@triton.jit
def _finalize_sums_kernel(
    partial_ptr,
    invstd1_ptr,
    weight1_ptr,
    invstd2_ptr,
    weight2_ptr,
    sum1_ptr,
    vec1_ptr,
    sum3_ptr,
    vec2_ptr,
    stats_ptr,
    CHANNELS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    INV_NHW: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.arange(0, BLOCK_TILES)
    mask = tile < NUM_TILES
    offsets = tile * CHANNELS + c
    plane = NUM_TILES * CHANNELS

    sum_where = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    sum_dot1 = tl.sum(tl.load(partial_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    sum_dot2 = tl.sum(tl.load(partial_ptr + plane * 2 + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

    invstd1 = tl.load(invstd1_ptr + c).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c).to(tl.float32)
    invstd2 = tl.load(invstd2_ptr + c).to(tl.float32)
    weight2 = tl.load(weight2_ptr + c).to(tl.float32)

    mean_where = _mul_rn_f32(sum_where, INV_NHW)
    dot1_mean = _mul_rn_f32(sum_dot1, INV_NHW)
    invstd1_sq = _mul_rn_f32(invstd1, invstd1)
    coeff1 = _mul_rn_f32(dot1_mean, invstd1_sq)
    scale1 = _mul_rn_f32(invstd1, weight1)
    vec1 = _mul_rn_f32(sum_dot1, invstd1)

    dot2_mean = _mul_rn_f32(sum_dot2, INV_NHW)
    invstd2_sq = _mul_rn_f32(invstd2, invstd2)
    coeff2 = _mul_rn_f32(dot2_mean, invstd2_sq)
    scale2 = _mul_rn_f32(invstd2, weight2)
    vec2 = _mul_rn_f32(sum_dot2, invstd2)

    tl.store(sum1_ptr + c, sum_where)
    tl.store(vec1_ptr + c, vec1)
    tl.store(sum3_ptr + c, sum_where)
    tl.store(vec2_ptr + c, vec2)
    tl.store(stats_ptr + c, mean_where)
    tl.store(stats_ptr + CHANNELS + c, coeff1)
    tl.store(stats_ptr + CHANNELS * 2 + c, scale1)
    tl.store(stats_ptr + CHANNELS * 3 + c, coeff2)
    tl.store(stats_ptr + CHANNELS * 4 + c, scale2)


@triton.jit
def _dense_epilogue_kernel(
    out0_ptr,
    arg4_ptr,
    mean1_ptr,
    arg8_ptr,
    mean2_ptr,
    stats_ptr,
    out1_ptr,
    out2_ptr,
    TOTAL: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    c = offsets % CHANNELS

    where_value = tl.load(out0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c, mask=active, other=0.0).to(tl.float32)
    x1 = tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    x2 = tl.load(arg8_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    mean_where = tl.load(stats_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff1 = tl.load(stats_ptr + CHANNELS + c, mask=active, other=0.0).to(tl.float32)
    scale1 = tl.load(stats_ptr + CHANNELS * 2 + c, mask=active, other=0.0).to(tl.float32)
    coeff2 = tl.load(stats_ptr + CHANNELS * 3 + c, mask=active, other=0.0).to(tl.float32)
    scale2 = tl.load(stats_ptr + CHANNELS * 4 + c, mask=active, other=0.0).to(tl.float32)

    centered1 = _sub_rn_f32(x1, mean1)
    correction1 = _mul_rn_f32(centered1, coeff1)
    tmp1 = _sub_rn_f32(where_value, correction1)
    tmp1 = _sub_rn_f32(tmp1, mean_where)
    dense1 = _mul_rn_f32(tmp1, scale1).to(tl.bfloat16)

    centered2 = _sub_rn_f32(x2, mean2)
    correction2 = _mul_rn_f32(centered2, coeff2)
    tmp2 = _sub_rn_f32(where_value, correction2)
    tmp2 = _sub_rn_f32(tmp2, mean_where)
    dense2 = _mul_rn_f32(tmp2, scale2).to(tl.bfloat16)

    tl.store(out1_ptr + offsets, dense1, mask=active)
    tl.store(out2_ptr + offsets, dense2, mask=active)


def _next_power_of_2(value):
    out = 1
    while out < int(value):
        out *= 2
    return out


def _shape_tuple(tensor):
    return tuple(int(dim) for dim in tensor.shape)


def _launch(
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
    out0,
    sum1,
    vec1,
    out1,
    sum3,
    vec2,
    out2,
    *,
    BLOCK_M: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    nhw = batch * hw
    total = batch * channels * hw
    num_tiles = triton.cdiv(nhw, BLOCK_M)
    block_tiles = _next_power_of_2(num_tiles)
    partial = torch.empty((3, num_tiles, channels), device=arg0_1.device, dtype=torch.float32)
    stats = torch.empty((5, channels), device=arg0_1.device, dtype=torch.float32)

    grid_partial = (num_tiles, triton.cdiv(channels, BLOCK_C))
    _partial_activation_sums_kernel[grid_partial](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg8_1,
        arg9_1,
        out0,
        partial,
        CHANNELS=channels,
        HEIGHT=height,
        WIDTH=width,
        HW=hw,
        NHW=nhw,
        NUM_TILES=num_tiles,
        STRIDE_N=arg0_1.stride(0),
        STRIDE_C=arg0_1.stride(1),
        STRIDE_H=arg0_1.stride(2),
        STRIDE_W=arg0_1.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _finalize_sums_kernel[(channels,)](
        partial,
        arg6_1,
        arg7_1,
        arg10_1,
        arg11_1,
        sum1,
        vec1,
        sum3,
        vec2,
        stats,
        CHANNELS=channels,
        NUM_TILES=num_tiles,
        INV_NHW=CAPTURED_INV_NHW,
        BLOCK_TILES=block_tiles,
        num_warps=8,
        num_stages=2,
    )

    _dense_epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        out0,
        arg4_1,
        arg5_1,
        arg8_1,
        arg9_1,
        stats,
        out1,
        out2,
        TOTAL=total,
        CHANNELS=channels,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=3,
    )


@oracle_impl(hardware="B200", point="25561d2e", BLOCK_M=128, BLOCK_C=16, EPILOGUE_BLOCK=256, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="63be79ac", BLOCK_M=128, BLOCK_C=16, EPILOGUE_BLOCK=256, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="f6f7ee3c", BLOCK_M=128, BLOCK_C=16, EPILOGUE_BLOCK=256, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
    num_warps: int,
    num_stages: int,
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
    ) = inputs
    shape = _shape_tuple(arg0_1)
    stride = tuple(int(dim) for dim in arg0_1.stride())
    channels = shape[1]

    out0 = torch.empty_strided(shape, stride, device=arg0_1.device, dtype=torch.float32)
    sum1 = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    vec1 = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    out1 = torch.empty_strided(shape, stride, device=arg0_1.device, dtype=torch.bfloat16)
    sum3 = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    vec2 = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    out2 = torch.empty_strided(shape, stride, device=arg0_1.device, dtype=torch.bfloat16)

    _launch(
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
        out0,
        sum1,
        vec1,
        out1,
        sum3,
        vec2,
        out2,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        EPILOGUE_BLOCK=EPILOGUE_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out0, sum1, vec1, out1, sum3, vec2, out2
