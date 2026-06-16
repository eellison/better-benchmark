"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full bf16 masked BatchNorm-backward tail by sharing the `where(arg0 <= 0, arg1, arg2)` producer across both channel reductions and then using the finalized per-channel sums to emit the returned sum vector, scale-gradient vector, and dense bf16 gradient tensor, whereas Inductor currently schedules the masked producer, sibling `sum([0, 2, 3])` reductions, and dependent dense epilogue as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a B200-tuned full-scope multi-output reduction template that keeps compatible channel reductions and their dependent tensor epilogue in one fused plan across both channels-last and contiguous NCHW layouts; the fix is SCHEDULER_FUSION: add scheduler/codegen support for shared masked channel reductions with finalized-scalar epilogues that write both vector and dense outputs."""

import torch
import triton
import triton.language as tl
from oracle_harness import oracle_impl


SCALE = 3.5189856312778704e-07
EPILOGUE_BLOCK = 1024


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _partial_reduce_nhwc_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    K_TOTAL: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets = k_offsets[:, None] * C + c_offsets[None, :]
    mask = (k_offsets[:, None] < K_TOTAL) & (c_offsets[None, :] < C)

    arg0 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0)
    arg2 = tl.load(arg2_ptr + offsets, mask=mask, other=0.0)
    arg3 = tl.load(arg3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    fill = tl.load(arg1_ptr)
    mean = tl.load(arg4_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)

    where_value = tl.where(arg0 <= 0.0, fill, arg2).to(tl.float32)
    centered = arg3 - mean[None, :]
    active_where = tl.where(mask, where_value, 0.0)
    active_dot = tl.where(mask, where_value * centered, 0.0)

    out_offsets = c_offsets * NUM_K_TILES + tl.program_id(1)
    tl.store(
        partial_sum_ptr + out_offsets,
        tl.sum(active_where, axis=0),
        mask=c_offsets < C,
    )
    tl.store(
        partial_dot_ptr + out_offsets,
        tl.sum(active_dot, axis=0),
        mask=c_offsets < C,
    )


@triton.jit
def _partial_reduce_nchw_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < K_TOTAL
    n = k // HW
    hw = k - n * HW
    offsets = n * C * HW + c * HW + hw

    arg0 = tl.load(arg0_ptr + offsets, mask=active, other=0.0)
    arg2 = tl.load(arg2_ptr + offsets, mask=active, other=0.0)
    arg3 = tl.load(arg3_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(arg1_ptr)
    mean = tl.load(arg4_ptr + c).to(tl.float32)

    where_value = tl.where(arg0 <= 0.0, fill, arg2).to(tl.float32)
    centered = arg3 - mean
    active_where = tl.where(active, where_value, 0.0)
    active_dot = tl.where(active, where_value * centered, 0.0)

    out_offset = c * NUM_K_TILES + tl.program_id(1)
    tl.store(partial_sum_ptr + out_offset, tl.sum(active_where, axis=0))
    tl.store(partial_dot_ptr + out_offset, tl.sum(active_dot, axis=0))


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    arg5_ptr,
    out_sum_ptr,
    sum2_ptr,
    out_vec_ptr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    mask = tile_offsets < NUM_K_TILES
    partial_offsets = c * NUM_K_TILES + tile_offsets

    sum1 = tl.sum(
        tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    sum2 = tl.sum(
        tl.load(partial_dot_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    weight = tl.load(arg5_ptr + c).to(tl.float32)

    tl.store(out_sum_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(out_vec_ptr + c, sum2 * weight)


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    if LAYOUT == "nhwc":
        c = offsets % C
    else:
        c = (offsets // HW) % C

    arg0 = tl.load(arg0_ptr + offsets, mask=active, other=0.0)
    arg2 = tl.load(arg2_ptr + offsets, mask=active, other=0.0)
    arg3 = tl.load(arg3_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(arg1_ptr)
    where_value = tl.where(arg0 <= 0.0, fill, arg2).to(tl.float32)

    mean = tl.load(arg4_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(arg5_ptr + c, mask=active, other=0.0).to(tl.float32)
    gamma = tl.load(arg6_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = arg3 - mean
    mean_term = sum1 * SCALE_VALUE
    mul_2 = sum2 * SCALE_VALUE
    mul_3 = invstd * invstd
    mul_4 = mul_2 * mul_3
    mul_5 = invstd * gamma
    mul_6 = centered * mul_4
    sub_1 = where_value - mul_6
    sub_2 = sub_1 - mean_term
    result = sub_2 * mul_5
    tl.store(out_ptr + offsets, result, mask=active)


@oracle_impl(hardware="B200", point="2cd54028", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="257ba105", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="67158ff7", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="75b1771b", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="dfd393d3", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="741c33e1", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="a5b95268", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="9d64073a", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="401c4c35", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="06c9e96f", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="125c973f", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="eaaaf33a", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="0df90ec4", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="1df68ba8", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="18f48531", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="8d74958e", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="992dca8a", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="4b502692", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="7ec4fd96", BLOCK_K=1024, BLOCK_C=8, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="146c34ef", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="3435a8f6", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="68768b75", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="399d5934", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="c85c640b", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="a38f76ef", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="845752f4", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="1b598f13", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="13f64ac0", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="2ba3e135", BLOCK_K=1024, BLOCK_C=8, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="008c68f0", BLOCK_K=1024, BLOCK_C=8, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="b65622a7", BLOCK_K=1024, BLOCK_C=8, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="ae27c8fd", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="52ba495d", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="7fe614dc", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="8d3f12fd", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=8)
@oracle_impl(hardware="B200", point="6d4a5d73", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="2605418e", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="c958bfc2", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="4e10c15b", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="144add86", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="ca4522ef", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="4841133c", BLOCK_K=1024, BLOCK_C=8, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="7effa056", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="3ab08c7e", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="76ea6bf0", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="f777c101", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="f7c89e98", BLOCK_K=1024, BLOCK_C=8, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="54ec885c", BLOCK_K=1024, BLOCK_C=8, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="779af19f", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=8)
@oracle_impl(hardware="B200", point="2d06f502", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=8)
@oracle_impl(hardware="B200", point="2621aaa7", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="c3020612", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="40e392d0", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="73d915c0", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="dcb5e067", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="5556e7e9", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="95bfb998", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="c0ea6855", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="927aa925", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="6424a3de", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="d8f6241e", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="23c01152", BLOCK_K=2048, BLOCK_C=1, LAYOUT="nchw", num_warps=8)
@oracle_impl(hardware="B200", point="652418c5", BLOCK_K=2048, BLOCK_C=1, LAYOUT="nchw", num_warps=8)
@oracle_impl(hardware="B200", point="3f125adf", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="48553d51", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="dc56431c", BLOCK_K=2048, BLOCK_C=1, LAYOUT="nchw", num_warps=8)
@oracle_impl(hardware="B200", point="b1b762bc", BLOCK_K=1024, BLOCK_C=1, LAYOUT="nchw", num_warps=4)
@oracle_impl(hardware="B200", point="e3c83a56", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="db481baa", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="e29d80d8", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="17b819ae", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="bc1103e2", BLOCK_K=2048, BLOCK_C=16, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="ab7835ac", BLOCK_K=1024, BLOCK_C=8, LAYOUT="nhwc", num_warps=8)
@oracle_impl(hardware="B200", point="0523f434", BLOCK_K=1024, BLOCK_C=8, LAYOUT="nhwc", num_warps=8)
def oracle_forward(inputs, *, BLOCK_K, BLOCK_C, LAYOUT, num_warps):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6 = inputs
    n, c, h, w = arg0.shape
    hw = h * w
    k_total = n * hw
    total = arg0.numel()
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    partial_sum = torch.empty((c, num_k_tiles), device=arg0.device, dtype=torch.float32)
    partial_dot = torch.empty((c, num_k_tiles), device=arg0.device, dtype=torch.float32)
    out_sum = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    sum2 = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    out_vec = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    out = torch.empty_strided(
        tuple(arg0.shape),
        tuple(arg0.stride()),
        device=arg0.device,
        dtype=torch.bfloat16,
    )

    if LAYOUT == "nhwc":
        _partial_reduce_nhwc_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            partial_sum,
            partial_dot,
            C=c,
            K_TOTAL=k_total,
            NUM_K_TILES=num_k_tiles,
            BLOCK_K=BLOCK_K,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
        )
    else:
        _partial_reduce_nchw_kernel[(c, num_k_tiles)](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            partial_sum,
            partial_dot,
            C=c,
            HW=hw,
            K_TOTAL=k_total,
            NUM_K_TILES=num_k_tiles,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
        )

    _finalize_kernel[(c,)](
        partial_sum,
        partial_dot,
        arg5,
        out_sum,
        sum2,
        out_vec,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        out_sum,
        sum2,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
        SCALE_VALUE=SCALE,
        LAYOUT=LAYOUT,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return out_sum, out_vec, out
