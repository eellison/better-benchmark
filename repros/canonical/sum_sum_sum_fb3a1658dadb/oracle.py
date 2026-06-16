"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete masked dual batch-norm-backward return tuple by sharing the bf16 add/ReLU-mask producer across the common `sum(where)` and both `sum(where * centered)` channel reductions, finalizing all four vector outputs, and writing both bf16 dense epilogues with the captured output strides; whereas Inductor lowers the masked producer, repeated sibling reductions, and dependent dense epilogues as separate generic pointwise/reduction regions; Inductor cannot do this today because its scheduler lacks a cooperative split-K multi-output reduction plan that coordinates shared channel partials with finalized-scalar tensor epilogues across contiguous and channels-last layouts; the fix is COOPERATIVE_SPLIT_K: add a layout-aware split-K channel-reduction template that shares compatible partials and sinks the dual BN-backward epilogues into the same planned schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


REDUCE_SCALE = 6.228077168367346e-07


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
def _source_value(
    x0_ptr,
    x1_ptr,
    mask_ptr,
    full_ptr,
    offsets0,
    offsets1,
    offsets_mask,
    active,
):
    x0 = tl.load(x0_ptr + offsets0, mask=active, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets1, mask=active, other=0.0).to(tl.float32)
    mask_value = tl.load(mask_ptr + offsets_mask, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    added = _f32_add(x0, x1).to(tl.bfloat16).to(tl.float32)
    return tl.where(mask_value <= 0.0, full_value, added)


@triton.jit
def _partial_reduce_kernel(
    x0_ptr,
    x1_ptr,
    mask_ptr,
    full_ptr,
    act0_ptr,
    mean0_ptr,
    act1_ptr,
    mean1_ptr,
    partial_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    NHW: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    X0_SN: tl.constexpr,
    X0_SC: tl.constexpr,
    X0_SH: tl.constexpr,
    X0_SW: tl.constexpr,
    X1_SN: tl.constexpr,
    X1_SC: tl.constexpr,
    X1_SH: tl.constexpr,
    X1_SW: tl.constexpr,
    XM_SN: tl.constexpr,
    XM_SC: tl.constexpr,
    XM_SH: tl.constexpr,
    XM_SW: tl.constexpr,
    A0_SN: tl.constexpr,
    A0_SC: tl.constexpr,
    A0_SH: tl.constexpr,
    A0_SW: tl.constexpr,
    A1_SN: tl.constexpr,
    A1_SC: tl.constexpr,
    A1_SH: tl.constexpr,
    A1_SW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cidx = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block = tl.program_id(1)
    k = block * BLOCK_K + tl.arange(0, BLOCK_K)
    n = k // (H * W)
    rem = k - n * (H * W)
    h = rem // W
    w = rem - h * W
    active = (k[:, None] < NHW) & (cidx[None, :] < C)

    offsets0 = n[:, None] * X0_SN + cidx[None, :] * X0_SC + h[:, None] * X0_SH + w[:, None] * X0_SW
    offsets1 = n[:, None] * X1_SN + cidx[None, :] * X1_SC + h[:, None] * X1_SH + w[:, None] * X1_SW
    offsets_mask = n[:, None] * XM_SN + cidx[None, :] * XM_SC + h[:, None] * XM_SH + w[:, None] * XM_SW
    offsets_a0 = n[:, None] * A0_SN + cidx[None, :] * A0_SC + h[:, None] * A0_SH + w[:, None] * A0_SW
    offsets_a1 = n[:, None] * A1_SN + cidx[None, :] * A1_SC + h[:, None] * A1_SH + w[:, None] * A1_SW

    source = _source_value(x0_ptr, x1_ptr, mask_ptr, full_ptr, offsets0, offsets1, offsets_mask, active)
    mean0 = tl.load(mean0_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    centered0 = _f32_sub(tl.load(act0_ptr + offsets_a0, mask=active, other=0.0).to(tl.float32), mean0[None, :])
    centered1 = _f32_sub(tl.load(act1_ptr + offsets_a1, mask=active, other=0.0).to(tl.float32), mean1[None, :])

    zeros = tl.full((BLOCK_K, BLOCK_C), 0.0, tl.float32)
    plane = C * NUM_BLOCKS
    partial_offsets = cidx * NUM_BLOCKS + block
    tl.store(partial_ptr + partial_offsets, tl.sum(tl.where(active, source, zeros), axis=0), mask=cidx < C)
    tl.store(
        partial_ptr + plane + partial_offsets,
        tl.sum(tl.where(active, _f32_mul(source, centered0), zeros), axis=0),
        mask=cidx < C,
    )
    tl.store(
        partial_ptr + plane * 2 + partial_offsets,
        tl.sum(tl.where(active, _f32_mul(source, centered1), zeros), axis=0),
        mask=cidx < C,
    )


@triton.jit
def _finalize_kernel(
    partial_ptr,
    gamma0_ptr,
    beta0_ptr,
    gamma1_ptr,
    beta1_ptr,
    stats_ptr,
    sum0_ptr,
    vec0_ptr,
    sum1_ptr,
    vec1_ptr,
    C: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    INV_NHW: tl.constexpr,
):
    c = tl.program_id(0)
    blocks = tl.arange(0, BLOCK_B)
    mask = blocks < NUM_BLOCKS
    plane = C * NUM_BLOCKS
    partial_offsets = c * NUM_BLOCKS + blocks
    sum_source = tl.sum(tl.load(partial_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    dot0 = tl.sum(tl.load(partial_ptr + plane + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    dot1 = tl.sum(tl.load(partial_ptr + plane * 2 + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

    gamma0 = tl.load(gamma0_ptr + c).to(tl.float32)
    beta0 = tl.load(beta0_ptr + c).to(tl.float32)
    gamma1 = tl.load(gamma1_ptr + c).to(tl.float32)
    beta1 = tl.load(beta1_ptr + c).to(tl.float32)

    mean_source = _f32_mul(sum_source, INV_NHW)
    dot0_mean = _f32_mul(dot0, INV_NHW)
    dot1_mean = _f32_mul(dot1, INV_NHW)
    coeff0 = _f32_mul(dot0_mean, _f32_mul(gamma0, gamma0))
    coeff1 = _f32_mul(dot1_mean, _f32_mul(gamma1, gamma1))
    scale0 = _f32_mul(gamma0, beta0)
    scale1 = _f32_mul(gamma1, beta1)

    tl.store(stats_ptr + c, mean_source)
    tl.store(stats_ptr + C + c, coeff0)
    tl.store(stats_ptr + C * 2 + c, scale0)
    tl.store(stats_ptr + C * 3 + c, coeff1)
    tl.store(stats_ptr + C * 4 + c, scale1)
    tl.store(sum0_ptr + c, sum_source)
    tl.store(vec0_ptr + c, _f32_mul(dot0, gamma0))
    tl.store(sum1_ptr + c, sum_source)
    tl.store(vec1_ptr + c, _f32_mul(dot1, gamma1))


@triton.jit
def _dense_epilogue_kernel(
    x0_ptr,
    x1_ptr,
    mask_ptr,
    full_ptr,
    act0_ptr,
    mean0_ptr,
    act1_ptr,
    mean1_ptr,
    stats_ptr,
    out0_ptr,
    out1_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    X0_SN: tl.constexpr,
    X0_SC: tl.constexpr,
    X0_SH: tl.constexpr,
    X0_SW: tl.constexpr,
    X1_SN: tl.constexpr,
    X1_SC: tl.constexpr,
    X1_SH: tl.constexpr,
    X1_SW: tl.constexpr,
    XM_SN: tl.constexpr,
    XM_SC: tl.constexpr,
    XM_SH: tl.constexpr,
    XM_SW: tl.constexpr,
    A0_SN: tl.constexpr,
    A0_SC: tl.constexpr,
    A0_SH: tl.constexpr,
    A0_SW: tl.constexpr,
    A1_SN: tl.constexpr,
    A1_SC: tl.constexpr,
    A1_SH: tl.constexpr,
    A1_SW: tl.constexpr,
    O0_SN: tl.constexpr,
    O0_SC: tl.constexpr,
    O0_SH: tl.constexpr,
    O0_SW: tl.constexpr,
    O1_SN: tl.constexpr,
    O1_SC: tl.constexpr,
    O1_SH: tl.constexpr,
    O1_SW: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    active = linear < TOTAL
    if LAYOUT == 1:
        c = linear % C
        t = linear // C
        w = t % W
        t = t // W
        h = t % H
        n = t // H
    else:
        w = linear % W
        t = linear // W
        h = t % H
        t = t // H
        c = t % C
        n = t // C

    offsets0 = n * X0_SN + c * X0_SC + h * X0_SH + w * X0_SW
    offsets1 = n * X1_SN + c * X1_SC + h * X1_SH + w * X1_SW
    offsets_mask = n * XM_SN + c * XM_SC + h * XM_SH + w * XM_SW
    offsets_a0 = n * A0_SN + c * A0_SC + h * A0_SH + w * A0_SW
    offsets_a1 = n * A1_SN + c * A1_SC + h * A1_SH + w * A1_SW
    offsets_o0 = n * O0_SN + c * O0_SC + h * O0_SH + w * O0_SW
    offsets_o1 = n * O1_SN + c * O1_SC + h * O1_SH + w * O1_SW

    source = _source_value(x0_ptr, x1_ptr, mask_ptr, full_ptr, offsets0, offsets1, offsets_mask, active)
    mean_source = tl.load(stats_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff0 = tl.load(stats_ptr + C + c, mask=active, other=0.0).to(tl.float32)
    scale0 = tl.load(stats_ptr + C * 2 + c, mask=active, other=0.0).to(tl.float32)
    coeff1 = tl.load(stats_ptr + C * 3 + c, mask=active, other=0.0).to(tl.float32)
    scale1 = tl.load(stats_ptr + C * 4 + c, mask=active, other=0.0).to(tl.float32)

    centered0 = _f32_sub(tl.load(act0_ptr + offsets_a0, mask=active, other=0.0).to(tl.float32), tl.load(mean0_ptr + c, mask=active, other=0.0).to(tl.float32))
    tmp0 = _f32_sub(_f32_sub(source, _f32_mul(centered0, coeff0)), mean_source)
    out0 = _f32_mul(tmp0, scale0).to(tl.bfloat16)

    centered1 = _f32_sub(tl.load(act1_ptr + offsets_a1, mask=active, other=0.0).to(tl.float32), tl.load(mean1_ptr + c, mask=active, other=0.0).to(tl.float32))
    tmp1 = _f32_sub(_f32_sub(source, _f32_mul(centered1, coeff1)), mean_source)
    out1 = _f32_mul(tmp1, scale1).to(tl.bfloat16)

    tl.store(out0_ptr + offsets_o0, out0, mask=active)
    tl.store(out1_ptr + offsets_o1, out1, mask=active)


def _next_power_of_2(value):
    return 1 << (value - 1).bit_length()


def _launch(inputs, *, BLOCK_K, BLOCK_C, BLOCK_E, num_warps_reduce, num_warps_finish, num_warps_epilogue):
    x0, x1, mask, full, act0, mean0, gamma0, beta0, act1, mean1, gamma1, beta1 = inputs
    n, c, h, w = x0.shape
    total = n * c * h * w
    nhw = n * h * w
    num_blocks = triton.cdiv(nhw, BLOCK_K)
    block_b = _next_power_of_2(num_blocks)

    partial = torch.empty((3, c, num_blocks), device=x0.device, dtype=torch.float32)
    stats = torch.empty((5, c), device=x0.device, dtype=torch.float32)
    sum0 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    vec0 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    sum1 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    vec1 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    out0 = torch.empty_strided(tuple(act0.shape), tuple(act0.stride()), device=act0.device, dtype=act0.dtype)
    out1 = torch.empty_strided(tuple(act1.shape), tuple(act1.stride()), device=act1.device, dtype=act1.dtype)

    x0s = x0.stride()
    x1s = x1.stride()
    masks = mask.stride()
    act0s = act0.stride()
    act1s = act1.stride()
    out0s = out0.stride()
    out1s = out1.stride()
    layout = 1 if out0s[1] == 1 else 0

    _partial_reduce_kernel[(triton.cdiv(c, BLOCK_C), num_blocks)](
        x0,
        x1,
        mask,
        full,
        act0,
        mean0,
        act1,
        mean1,
        partial,
        C=c,
        H=h,
        W=w,
        NHW=nhw,
        NUM_BLOCKS=num_blocks,
        X0_SN=x0s[0],
        X0_SC=x0s[1],
        X0_SH=x0s[2],
        X0_SW=x0s[3],
        X1_SN=x1s[0],
        X1_SC=x1s[1],
        X1_SH=x1s[2],
        X1_SW=x1s[3],
        XM_SN=masks[0],
        XM_SC=masks[1],
        XM_SH=masks[2],
        XM_SW=masks[3],
        A0_SN=act0s[0],
        A0_SC=act0s[1],
        A0_SH=act0s[2],
        A0_SW=act0s[3],
        A1_SN=act1s[0],
        A1_SC=act1s[1],
        A1_SH=act1s[2],
        A1_SW=act1s[3],
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_reduce,
        num_stages=3,
    )
    _finalize_kernel[(c,)](
        partial,
        gamma0,
        beta0,
        gamma1,
        beta1,
        stats,
        sum0,
        vec0,
        sum1,
        vec1,
        C=c,
        NUM_BLOCKS=num_blocks,
        BLOCK_B=block_b,
        INV_NHW=REDUCE_SCALE,
        num_warps=num_warps_finish,
        num_stages=3,
    )
    _dense_epilogue_kernel[(triton.cdiv(total, BLOCK_E),)](
        x0,
        x1,
        mask,
        full,
        act0,
        mean0,
        act1,
        mean1,
        stats,
        out0,
        out1,
        TOTAL=total,
        C=c,
        H=h,
        W=w,
        X0_SN=x0s[0],
        X0_SC=x0s[1],
        X0_SH=x0s[2],
        X0_SW=x0s[3],
        X1_SN=x1s[0],
        X1_SC=x1s[1],
        X1_SH=x1s[2],
        X1_SW=x1s[3],
        XM_SN=masks[0],
        XM_SC=masks[1],
        XM_SH=masks[2],
        XM_SW=masks[3],
        A0_SN=act0s[0],
        A0_SC=act0s[1],
        A0_SH=act0s[2],
        A0_SW=act0s[3],
        A1_SN=act1s[0],
        A1_SC=act1s[1],
        A1_SH=act1s[2],
        A1_SW=act1s[3],
        O0_SN=out0s[0],
        O0_SC=out0s[1],
        O0_SH=out0s[2],
        O0_SW=out0s[3],
        O1_SN=out1s[0],
        O1_SC=out1s[1],
        O1_SH=out1s[2],
        O1_SW=out1s[3],
        LAYOUT=layout,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps_epilogue,
        num_stages=3,
    )
    return sum0, vec0, out0, sum1, vec1, out1


@oracle_impl(hardware="B200", point="7a67de76", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=8, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="32659d96", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="a05f5297", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="ee822c25", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="947f7696", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="4ff6585f", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="123127af", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="d2c78c03", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="b09bc4e6", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="db75ae04", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="45fc19b9", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="1f5f432a", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="f3056617", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="5ca11463", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="5ebdcf4f", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="0b10d4fe", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="5edb39f6", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="6880ff5a", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="0454b2c3", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="80c40bdb", BLOCK_K=512, BLOCK_C=4, BLOCK_E=256, num_warps_reduce=8, num_warps_finish=4, num_warps_epilogue=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_K,
    BLOCK_C,
    BLOCK_E,
    num_warps_reduce,
    num_warps_finish,
    num_warps_epilogue,
):
    return _launch(
        inputs,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        BLOCK_E=BLOCK_E,
        num_warps_reduce=num_warps_reduce,
        num_warps_finish=num_warps_finish,
        num_warps_epilogue=num_warps_epilogue,
    )
