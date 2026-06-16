"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 SqueezeNet dual slice/where/sum scope by streaming the shared channels-last `[N,H,W]` domain once per channel tile, materializing both returned bf16 `where` tensors, and co-reducing their per-channel sums before the required bf16-to-f32 vector return. Inductor currently schedules the two masked slice producers and sibling channel reductions as generic regions even though they share axes, output layout, scalar fill, and the same wide input tensor; it cannot form one full-scope multi-output reduction/store template for independent channel halves and masks. The fix is SCHEDULER_FUSION: teach the scheduler to fuse compatible sibling masked reductions and returned producers into one multi-accumulator channels-last plan with separate epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 32
C = 256
IN_C = 512
SLICE_OFFSET = 256
H = 13
W = 13
HW = H * W
K_TOTAL = N * HW


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _partial_where_sum_kernel(
    wide_ptr,
    right_mask_ptr,
    fill_ptr,
    left_mask_ptr,
    right_out_ptr,
    left_out_ptr,
    partial_right_ptr,
    partial_left_ptr,
    NUM_K_BLOCKS: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    C_: tl.constexpr,
    IN_C_: tl.constexpr,
    SLICE_OFFSET_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = (k[:, None] < K_TOTAL_) & (c[None, :] < C_)

    narrow_offsets = k[:, None] * C_ + c[None, :]
    left_offsets = k[:, None] * IN_C_ + c[None, :]
    right_offsets = left_offsets + SLICE_OFFSET_

    fill = tl.load(fill_ptr)
    right_mask = tl.load(right_mask_ptr + narrow_offsets, mask=active, other=0)
    left_mask = tl.load(left_mask_ptr + narrow_offsets, mask=active, other=0)
    right_src = tl.load(wide_ptr + right_offsets, mask=active, other=0.0)
    left_src = tl.load(wide_ptr + left_offsets, mask=active, other=0.0)

    right = tl.where(right_mask, fill, right_src)
    left = tl.where(left_mask, fill, left_src)
    tl.store(right_out_ptr + narrow_offsets, right, mask=active)
    tl.store(left_out_ptr + narrow_offsets, left, mask=active)

    right_sum = tl.sum(tl.where(active, right.to(tl.float32), 0.0), axis=0)
    left_sum = tl.sum(tl.where(active, left.to(tl.float32), 0.0), axis=0)
    partial_offsets = c * NUM_K_BLOCKS + tl.program_id(1)
    tl.store(partial_right_ptr + partial_offsets, right_sum, mask=c < C_)
    tl.store(partial_left_ptr + partial_offsets, left_sum, mask=c < C_)


@triton.jit
def _finalize_kernel(
    partial_right_ptr,
    partial_left_ptr,
    right_sum_out_ptr,
    left_sum_out_ptr,
    NUM_K_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    active = (block_offsets[:, None] < NUM_K_BLOCKS) & (c[None, :] < C_)
    offsets = c[None, :] * NUM_K_BLOCKS + block_offsets[:, None]

    right = tl.sum(
        tl.load(partial_right_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    left = tl.sum(
        tl.load(partial_left_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    tl.store(right_sum_out_ptr + c, right, mask=c < C_)
    tl.store(left_sum_out_ptr + c, left, mask=c < C_)


@oracle_impl(
    hardware="B200",
    point="220691ab",
    BLOCK_K=1024,
    BLOCK_C=16,
    reduce_warps=8,
    final_warps=1,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    reduce_warps: int,
    final_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    num_k_blocks = triton.cdiv(K_TOTAL, BLOCK_K)

    right_out = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    left_out = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    right_sum = torch.empty_strided((C,), (1,), device=arg0_1.device, dtype=torch.float32)
    left_sum = torch.empty_strided((C,), (1,), device=arg0_1.device, dtype=torch.float32)
    partial_right = torch.empty_strided(
        (C, num_k_blocks),
        (num_k_blocks, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    partial_left = torch.empty_like(partial_right)

    _partial_where_sum_kernel[(triton.cdiv(C, BLOCK_C), num_k_blocks)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        right_out,
        left_out,
        partial_right,
        partial_left,
        NUM_K_BLOCKS=num_k_blocks,
        K_TOTAL_=K_TOTAL,
        C_=C,
        IN_C_=IN_C,
        SLICE_OFFSET_=SLICE_OFFSET,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=4,
    )
    _finalize_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial_right,
        partial_left,
        right_sum,
        left_sum,
        NUM_K_BLOCKS=num_k_blocks,
        C_=C,
        BLOCK_BLOCKS=_next_power_of_2(num_k_blocks),
        BLOCK_C=BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )
    return right_out, right_sum, left_out, left_sum
