"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 masked batch-norm-backward fragment by sharing the `sum(where)` and `sum(where * centered)` channel reductions, returning both f32 vectors, and emitting the dependent bf16 tensor epilogue with the captured layout, whereas Inductor lowers the masked add producer, sibling reductions, vector epilogue, and full-tensor epilogue as separate generic reduction/pointwise regions; Inductor cannot do this today because its scheduler/codegen lacks a full-scope channel-reduction template that keeps compatible reductions and their dependent BN-backward epilogue together across contiguous and channels-last layouts; the fix is SCHEDULER_FUSION: add a guarded masked channel-reduction schedule that shares the reduction producer and sinks the vector/tensor epilogues into one plan while preserving Inductor's f32 fused add/where and final bf16 output cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


REDUCE_SCALE = 0.0001220703125


@triton.jit
def _partial_reduce_cl_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    partial_sum_ptr,
    partial_prod_ptr,
    TOTAL_K: tl.constexpr,
    NUM_TILES: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S0N: tl.constexpr,
    S0C: tl.constexpr,
    S0H: tl.constexpr,
    S0W: tl.constexpr,
    S1N: tl.constexpr,
    S1C: tl.constexpr,
    S1H: tl.constexpr,
    S1W: tl.constexpr,
    S2N: tl.constexpr,
    S2C: tl.constexpr,
    S2H: tl.constexpr,
    S2W: tl.constexpr,
    S4N: tl.constexpr,
    S4C: tl.constexpr,
    S4H: tl.constexpr,
    S4W: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = (k[:, None] < TOTAL_K) & (channels[None, :] < C)

    hw = k % (H * W)
    n = k // (H * W)
    h = hw // W
    w = hw - h * W

    offs0 = n[:, None] * S0N + channels[None, :] * S0C + h[:, None] * S0H + w[:, None] * S0W
    offs1 = n[:, None] * S1N + channels[None, :] * S1C + h[:, None] * S1H + w[:, None] * S1W
    offs2 = n[:, None] * S2N + channels[None, :] * S2C + h[:, None] * S2H + w[:, None] * S2W
    offs4 = n[:, None] * S4N + channels[None, :] * S4C + h[:, None] * S4H + w[:, None] * S4W

    lhs = tl.load(arg0_ptr + offs0, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offs1, mask=active, other=0.0).to(tl.float32)
    gate = tl.load(arg2_ptr + offs2, mask=active, other=0.0).to(tl.float32) <= 0.0
    fill = tl.load(arg3_ptr).to(tl.float32)
    add_value = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
    where_value = tl.where(gate, fill, add_value)
    where_value = tl.where(active, where_value, 0.0)

    mean = tl.load(arg5_ptr + channels, mask=channels < C, other=0.0).to(tl.float32)
    centered = tl.load(arg4_ptr + offs4, mask=active, other=0.0).to(tl.float32) - mean[None, :]
    prod = tl.where(active, where_value * centered, 0.0)

    out_offsets = channels * NUM_TILES + tl.program_id(1)
    channel_mask = channels < C
    tl.store(partial_sum_ptr + out_offsets, tl.sum(where_value, axis=0), mask=channel_mask)
    tl.store(partial_prod_ptr + out_offsets, tl.sum(prod, axis=0), mask=channel_mask)


@triton.jit
def _partial_reduce_nchw_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    partial_sum_ptr,
    partial_prod_ptr,
    TOTAL_K: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S0N: tl.constexpr,
    S0C: tl.constexpr,
    S0H: tl.constexpr,
    S0W: tl.constexpr,
    S1N: tl.constexpr,
    S1C: tl.constexpr,
    S1H: tl.constexpr,
    S1W: tl.constexpr,
    S2N: tl.constexpr,
    S2C: tl.constexpr,
    S2H: tl.constexpr,
    S2W: tl.constexpr,
    S4N: tl.constexpr,
    S4C: tl.constexpr,
    S4H: tl.constexpr,
    S4W: tl.constexpr,
    BLOCK_K: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    r_offsets = tl.arange(0, R_BLOCK)[None, :]
    sum_acc = tl.full((1, R_BLOCK), 0.0, tl.float32)
    prod_acc = tl.full((1, R_BLOCK), 0.0, tl.float32)
    mean = tl.load(arg5_ptr + c).to(tl.float32)
    fill = tl.load(arg3_ptr).to(tl.float32)

    for r_start in tl.range(0, BLOCK_K, R_BLOCK):
        k = tile * BLOCK_K + r_start + r_offsets
        active = k < TOTAL_K

        hw = k % (H * W)
        n = k // (H * W)
        h = hw // W
        w = hw - h * W

        offs0 = n * S0N + c * S0C + h * S0H + w * S0W
        offs1 = n * S1N + c * S1C + h * S1H + w * S1W
        offs2 = n * S2N + c * S2C + h * S2H + w * S2W
        offs4 = n * S4N + c * S4C + h * S4H + w * S4W

        lhs = tl.load(arg0_ptr + offs0, mask=active, other=0.0).to(tl.float32)
        rhs = tl.load(arg1_ptr + offs1, mask=active, other=0.0).to(tl.float32)
        gate = tl.load(arg2_ptr + offs2, mask=active, other=0.0).to(tl.float32) <= 0.0
        add_value = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
        where_value = tl.where(gate, fill, add_value)
        where_value = tl.where(active, where_value, 0.0)
        centered = tl.load(arg4_ptr + offs4, mask=active, other=0.0).to(tl.float32) - mean
        prod = tl.where(active, where_value * centered, 0.0)

        sum_acc += where_value
        prod_acc += prod

    out_offset = tile * C + c
    one = tl.arange(0, 1)
    tl.store(partial_sum_ptr + out_offset + one, tl.sum(sum_acc, axis=1))
    tl.store(partial_prod_ptr + out_offset + one, tl.sum(prod_acc, axis=1))


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_prod_ptr,
    invstd_ptr,
    sum_out_ptr,
    vec_out_ptr,
    sum2_ptr,
    NUM_TILES: tl.constexpr,
    C: tl.constexpr,
    NCHW_LAYOUT: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)[None, :]
    mask = tiles < NUM_TILES
    if NCHW_LAYOUT:
        offsets = tiles * C + c
    else:
        offsets = c * NUM_TILES + tiles

    sum_vals = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    prod_vals = tl.load(partial_prod_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum1 = tl.sum(sum_vals, axis=1)
    sum2 = tl.sum(prod_vals, axis=1)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    one = tl.arange(0, 1)
    tl.store(sum_out_ptr + c + one, sum1)
    tl.store(sum2_ptr + c + one, sum2)
    tl.store(vec_out_ptr + c + one, sum2 * invstd)


@triton.jit
def _epilogue_cl_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    arg7_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    NUMEL: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S0N: tl.constexpr,
    S0C: tl.constexpr,
    S0H: tl.constexpr,
    S0W: tl.constexpr,
    S1N: tl.constexpr,
    S1C: tl.constexpr,
    S1H: tl.constexpr,
    S1W: tl.constexpr,
    S2N: tl.constexpr,
    S2C: tl.constexpr,
    S2H: tl.constexpr,
    S2W: tl.constexpr,
    S4N: tl.constexpr,
    S4C: tl.constexpr,
    S4H: tl.constexpr,
    S4W: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < NUMEL

    c = offsets % C
    w = (offsets // C) % W
    h = (offsets // (C * W)) % H
    n = offsets // (C * H * W)

    offs0 = n * S0N + c * S0C + h * S0H + w * S0W
    offs1 = n * S1N + c * S1C + h * S1H + w * S1W
    offs2 = n * S2N + c * S2C + h * S2H + w * S2W
    offs4 = n * S4N + c * S4C + h * S4H + w * S4W

    lhs = tl.load(arg0_ptr + offs0, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offs1, mask=active, other=0.0).to(tl.float32)
    gate = tl.load(arg2_ptr + offs2, mask=active, other=0.0).to(tl.float32) <= 0.0
    fill = tl.load(arg3_ptr).to(tl.float32)
    add_value = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
    where_value = tl.where(gate, fill, add_value)

    mean = tl.load(arg5_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered = tl.load(arg4_ptr + offs4, mask=active, other=0.0).to(tl.float32) - mean
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(arg6_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(arg7_ptr + c, mask=active, other=0.0).to(tl.float32)

    variance_term = (sum2 * SCALE) * (invstd * invstd)
    mean_term = sum1 * SCALE
    out = ((where_value - centered * variance_term) - mean_term) * (invstd * weight)
    tl.store(out_ptr + offsets, out, mask=active)


@triton.jit
def _epilogue_nchw_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    arg7_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    NUMEL: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S0N: tl.constexpr,
    S0C: tl.constexpr,
    S0H: tl.constexpr,
    S0W: tl.constexpr,
    S1N: tl.constexpr,
    S1C: tl.constexpr,
    S1H: tl.constexpr,
    S1W: tl.constexpr,
    S2N: tl.constexpr,
    S2C: tl.constexpr,
    S2H: tl.constexpr,
    S2W: tl.constexpr,
    S4N: tl.constexpr,
    S4C: tl.constexpr,
    S4H: tl.constexpr,
    S4W: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < NUMEL

    w = offsets % W
    h = (offsets // W) % H
    c = (offsets // (H * W)) % C
    n = offsets // (C * H * W)

    offs0 = n * S0N + c * S0C + h * S0H + w * S0W
    offs1 = n * S1N + c * S1C + h * S1H + w * S1W
    offs2 = n * S2N + c * S2C + h * S2H + w * S2W
    offs4 = n * S4N + c * S4C + h * S4H + w * S4W

    lhs = tl.load(arg0_ptr + offs0, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offs1, mask=active, other=0.0).to(tl.float32)
    gate = tl.load(arg2_ptr + offs2, mask=active, other=0.0).to(tl.float32) <= 0.0
    fill = tl.load(arg3_ptr).to(tl.float32)
    add_value = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
    where_value = tl.where(gate, fill, add_value)

    mean = tl.load(arg5_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered = tl.load(arg4_ptr + offs4, mask=active, other=0.0).to(tl.float32) - mean
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(arg6_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(arg7_ptr + c, mask=active, other=0.0).to(tl.float32)

    variance_term = (sum2 * SCALE) * (invstd * invstd)
    mean_term = sum1 * SCALE
    out = ((where_value - centered * variance_term) - mean_term) * (invstd * weight)
    tl.store(out_ptr + offsets, out, mask=active)


def _launch(inputs, *, BLOCK_K: int, BLOCK_C: int, EPILOGUE_BLOCK: int, num_warps: int):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 = inputs
    n = int(arg4.shape[0])
    c = int(arg4.shape[1])
    h = int(arg4.shape[2])
    w = int(arg4.shape[3])
    total_k = n * h * w
    num_tiles = triton.cdiv(total_k, BLOCK_K)

    sum1 = torch.empty((c,), device=arg4.device, dtype=torch.float32)
    sum2 = torch.empty((c,), device=arg4.device, dtype=torch.float32)
    vec_out = torch.empty((c,), device=arg4.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(arg4.shape), tuple(arg4.stride()), device=arg4.device, dtype=torch.bfloat16)
    partial_sum = torch.empty((c, num_tiles), device=arg4.device, dtype=torch.float32)
    partial_prod = torch.empty((c, num_tiles), device=arg4.device, dtype=torch.float32)

    s0 = tuple(int(s) for s in arg0.stride())
    s1 = tuple(int(s) for s in arg1.stride())
    s2 = tuple(int(s) for s in arg2.stride())
    s4 = tuple(int(s) for s in arg4.stride())
    nchw = arg4.stride(1) != 1

    if not nchw:
        _partial_reduce_cl_kernel[(triton.cdiv(c, BLOCK_C), num_tiles)](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            partial_sum,
            partial_prod,
            TOTAL_K=total_k,
            NUM_TILES=num_tiles,
            C=c,
            H=h,
            W=w,
            S0N=s0[0],
            S0C=s0[1],
            S0H=s0[2],
            S0W=s0[3],
            S1N=s1[0],
            S1C=s1[1],
            S1H=s1[2],
            S1W=s1[3],
            S2N=s2[0],
            S2C=s2[1],
            S2H=s2[2],
            S2W=s2[3],
            S4N=s4[0],
            S4C=s4[1],
            S4H=s4[2],
            S4W=s4[3],
            BLOCK_K=BLOCK_K,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        _partial_reduce_nchw_kernel[(c, num_tiles)](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            partial_sum,
            partial_prod,
            TOTAL_K=total_k,
            C=c,
            H=h,
            W=w,
            S0N=s0[0],
            S0C=s0[1],
            S0H=s0[2],
            S0W=s0[3],
            S1N=s1[0],
            S1C=s1[1],
            S1H=s1[2],
            S1W=s1[3],
            S2N=s2[0],
            S2C=s2[1],
            S2H=s2[2],
            S2W=s2[3],
            S4N=s4[0],
            S4C=s4[1],
            S4H=s4[2],
            S4W=s4[3],
            BLOCK_K=BLOCK_K,
            R_BLOCK=2048,
            num_warps=16,
            num_stages=3,
        )

    _finalize_kernel[(c,)](
        partial_sum,
        partial_prod,
        arg6,
        sum1,
        vec_out,
        sum2,
        NUM_TILES=num_tiles,
        C=c,
        NCHW_LAYOUT=nchw,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=num_warps,
        num_stages=3,
    )

    epilogue_grid = (triton.cdiv(n * c * h * w, EPILOGUE_BLOCK),)
    epilogue = _epilogue_nchw_kernel if nchw else _epilogue_cl_kernel
    epilogue[epilogue_grid](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        sum1,
        sum2,
        out,
        NUMEL=n * c * h * w,
        C=c,
        H=h,
        W=w,
        S0N=s0[0],
        S0C=s0[1],
        S0H=s0[2],
        S0W=s0[3],
        S1N=s1[0],
        S1C=s1[1],
        S1H=s1[2],
        S1W=s1[3],
        S2N=s2[0],
        S2C=s2[1],
        S2H=s2[2],
        S2W=s2[3],
        S4N=s4[0],
        S4C=s4[1],
        S4H=s4[2],
        S4W=s4[3],
        SCALE=REDUCE_SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=3,
    )

    return sum1, vec_out, out


@oracle_impl(hardware="B200", point="33ee22dc", BLOCK_K=256, BLOCK_C=16, EPILOGUE_BLOCK=256, num_warps=8)
@oracle_impl(hardware="B200", point="cd62c4c8", BLOCK_K=512, BLOCK_C=16, EPILOGUE_BLOCK=256, num_warps=8)
@oracle_impl(hardware="B200", point="1f3fcf29", BLOCK_K=512, BLOCK_C=16, EPILOGUE_BLOCK=256, num_warps=8)
@oracle_impl(hardware="B200", point="2f0e8753", BLOCK_K=8192, BLOCK_C=16, EPILOGUE_BLOCK=256, num_warps=2)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, EPILOGUE_BLOCK: int, num_warps: int):
    return _launch(inputs, BLOCK_K=BLOCK_K, BLOCK_C=BLOCK_C, EPILOGUE_BLOCK=EPILOGUE_BLOCK, num_warps=num_warps)
