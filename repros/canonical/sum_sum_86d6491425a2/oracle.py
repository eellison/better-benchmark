"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 SqueezeNet dual slice/where/sum scope by materializing both returned channels-last `where` tensors and accumulating the two sibling channel sums from the same tiled pass over `[N,H,W,C]`, returning the converted sums from fp32 accumulators to match the compiled path's measured numerics; whereas Inductor schedules the visible `where` outputs and sibling reductions as generic work around materialized producers. Inductor cannot do this today because its reduction scheduler does not form one full-scope multi-output template for compatible sibling reductions that share axes/output layout but read different channel halves and masks while also sinking live side-output stores. The fix is SCHEDULER_FUSION: teach Inductor to fuse sibling masked channel reductions with their required side-output materialization and exact reduction accumulation boundary."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 32
C = 192
FULL_C = 384
H = 13
W = 13
HW = H * W
R = N * HW


@triton.jit
def _where_sum_partials_kernel(
    input_ptr,
    mask_right_ptr,
    fill_ptr,
    mask_left_ptr,
    out_right_ptr,
    out_left_ptr,
    partial_ptr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NUM_R_TILES: tl.constexpr,
    R_: tl.constexpr,
    C_: tl.constexpr,
    FULL_C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    active = (r[:, None] < R_) & (c[None, :] < C_)

    n = r // HW_
    hw = r - n * HW_
    h = hw // W_
    w = hw - h * W_

    input_base = n[:, None] * (FULL_C_ * HW_) + h[:, None] * (W_ * FULL_C_) + w[:, None] * FULL_C_ + c[None, :]
    out_base = n[:, None] * (C_ * HW_) + h[:, None] * (W_ * C_) + w[:, None] * C_ + c[None, :]

    fill = tl.load(fill_ptr)
    right = tl.load(input_ptr + input_base + C_, mask=active, other=0.0)
    left = tl.load(input_ptr + input_base, mask=active, other=0.0)
    right_mask = tl.load(mask_right_ptr + out_base, mask=active, other=0) != 0
    left_mask = tl.load(mask_left_ptr + out_base, mask=active, other=0) != 0

    right_where = tl.where(right_mask, fill, right)
    left_where = tl.where(left_mask, fill, left)
    tl.store(out_right_ptr + out_base, right_where, mask=active)
    tl.store(out_left_ptr + out_base, left_where, mask=active)

    right_sum = tl.sum(tl.where(active, right_where.to(tl.float32), 0.0), axis=0)
    left_sum = tl.sum(tl.where(active, left_where.to(tl.float32), 0.0), axis=0)
    partial_offsets = tl.program_id(1) * C_ + c
    channel_mask = c < C_
    tl.store(partial_ptr + partial_offsets, right_sum, mask=channel_mask)
    tl.store(partial_ptr + NUM_R_TILES * C_ + partial_offsets, left_sum, mask=channel_mask)


@triton.jit
def _where_sum_finalize_kernel(
    partial_ptr,
    out_right_sum_ptr,
    out_left_sum_ptr,
    BLOCK_C: tl.constexpr,
    NUM_R_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    active = (tiles[:, None] < NUM_R_TILES) & (c[None, :] < C_)
    offsets = tiles[:, None] * C_ + c[None, :]

    right_partials = tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    left_partials = tl.load(partial_ptr + NUM_R_TILES * C_ + offsets, mask=active, other=0.0).to(tl.float32)
    right_sum = tl.sum(right_partials, axis=0)
    left_sum = tl.sum(left_partials, axis=0)

    channel_mask = c < C_
    tl.store(out_right_sum_ptr + c, right_sum, mask=channel_mask)
    tl.store(out_left_sum_ptr + c, left_sum, mask=channel_mask)


# d987ff10: (T([32,384,13,13], bf16, channels-last), T([32,192,13,13], b8, channels-last), T([], bf16), T([32,192,13,13], b8, channels-last))
@oracle_impl(hardware="B200", point="d987ff10", BLOCK_R=256, BLOCK_C=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    device = arg0_1.device
    num_r_tiles = triton.cdiv(R, BLOCK_R)
    block_tiles = triton.next_power_of_2(num_r_tiles)

    where = torch.empty_strided((N, C, H, W), (C * HW, 1, W * C, C), device=device, dtype=torch.bfloat16)
    where_1 = torch.empty_strided((N, C, H, W), (C * HW, 1, W * C, C), device=device, dtype=torch.bfloat16)
    sum_1 = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    sum_2 = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    partial = torch.empty_strided((2, num_r_tiles, C), (num_r_tiles * C, C, 1), device=device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), num_r_tiles)
    _where_sum_partials_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        where,
        where_1,
        partial,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        NUM_R_TILES=num_r_tiles,
        R_=R,
        C_=C,
        FULL_C_=FULL_C,
        HW_=HW,
        W_=W,
        num_warps=num_warps,
    )
    _where_sum_finalize_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial,
        sum_1,
        sum_2,
        BLOCK_C=BLOCK_C,
        NUM_R_TILES=num_r_tiles,
        BLOCK_TILES=block_tiles,
        C_=C,
        num_warps=1,
    )
    return where, sum_1, where_1, sum_2
