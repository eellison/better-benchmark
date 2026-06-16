"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SqueezeNet bf16 dual slice/where/sum scope by materializing both returned channels-last `where` tensors while co-reducing the lower and upper channel halves in one tiled Triton pass into the compiled-path f32 vector sums, whereas Inductor schedules the sibling masked materializations and reductions as generic regions over the shared N,H,W iteration space; Inductor cannot do this today because its scheduler/codegen does not form one full-scope multi-output plan that shares compatible channels-last producers while preserving returned dense outputs and the f32 reduction outputs Inductor emits for this capture; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse compatible sibling masked reductions with their dense materialized outputs into one multi-accumulator channels-last template."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _dual_where_partial_kernel(
    input_ptr,
    mask_right_ptr,
    fill_ptr,
    mask_left_ptr,
    out_right_ptr,
    out_left_ptr,
    partial_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    INPUT_STRIDE_N: tl.constexpr,
    INPUT_STRIDE_H: tl.constexpr,
    INPUT_STRIDE_W: tl.constexpr,
    MASK_STRIDE_N: tl.constexpr,
    MASK_STRIDE_H: tl.constexpr,
    MASK_STRIDE_W: tl.constexpr,
    RIGHT_C_OFFSET: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active_k = k_offsets < K_TOTAL
    active_c = c_offsets < C
    active = active_k[:, None] & active_c[None, :]

    n = k_offsets // HW
    spatial = k_offsets - n * HW
    h = spatial // W
    w = spatial - h * W

    input_offsets = (
        n[:, None] * INPUT_STRIDE_N
        + h[:, None] * INPUT_STRIDE_H
        + w[:, None] * INPUT_STRIDE_W
        + c_offsets[None, :]
    )
    out_offsets = (
        n[:, None] * MASK_STRIDE_N
        + h[:, None] * MASK_STRIDE_H
        + w[:, None] * MASK_STRIDE_W
        + c_offsets[None, :]
    )

    fill = tl.load(fill_ptr)
    right_values = tl.load(
        input_ptr + input_offsets + RIGHT_C_OFFSET,
        mask=active,
        other=0.0,
    )
    left_values = tl.load(input_ptr + input_offsets, mask=active, other=0.0)
    right_mask = tl.load(mask_right_ptr + out_offsets, mask=active, other=0)
    left_mask = tl.load(mask_left_ptr + out_offsets, mask=active, other=0)

    right_where = tl.where(right_mask, fill, right_values)
    left_where = tl.where(left_mask, fill, left_values)
    tl.store(out_right_ptr + out_offsets, right_where, mask=active)
    tl.store(out_left_ptr + out_offsets, left_where, mask=active)

    right_sum = tl.sum(
        tl.where(active, right_where.to(tl.float32), 0.0),
        axis=0,
    )
    left_sum = tl.sum(
        tl.where(active, left_where.to(tl.float32), 0.0),
        axis=0,
    )
    partial_offsets = c_offsets * NUM_K_BLOCKS + tl.program_id(1)
    tl.store(partial_ptr + partial_offsets, right_sum, mask=active_c)
    tl.store(
        partial_ptr + C * NUM_K_BLOCKS + partial_offsets,
        left_sum,
        mask=active_c,
    )


@triton.jit
def _dual_where_final_kernel(
    partial_ptr,
    out_right_sum_ptr,
    out_left_sum_ptr,
    C: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    active = (block_offsets[:, None] < NUM_K_BLOCKS) & (c_offsets[None, :] < C)
    partial_offsets = c_offsets[None, :] * NUM_K_BLOCKS + block_offsets[:, None]

    right_partials = tl.load(partial_ptr + partial_offsets, mask=active, other=0.0)
    left_partials = tl.load(
        partial_ptr + C * NUM_K_BLOCKS + partial_offsets,
        mask=active,
        other=0.0,
    )
    right_sum = tl.sum(right_partials.to(tl.float32), axis=0)
    left_sum = tl.sum(left_partials.to(tl.float32), axis=0)

    out_mask = c_offsets < C
    tl.store(out_right_sum_ptr + c_offsets, right_sum, mask=out_mask)
    tl.store(out_left_sum_ptr + c_offsets, left_sum, mask=out_mask)


# 400df623: torchbench squeezenet1_1 train, bf16 [32, 256, 27, 27] channels-last input.
@oracle_impl(hardware="B200", point="400df623", BLOCK_K=128, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, FINAL_BLOCK_C: int, num_warps: int):
    arg0, arg1, arg2, arg3 = inputs

    n = 32
    c = 128
    h = 27
    w = 27
    hw = h * w
    k_total = n * hw
    num_k_blocks = triton.cdiv(k_total, BLOCK_K)
    block_blocks = _next_power_of_2(num_k_blocks)

    out_right = torch.empty_strided(
        (n, c, h, w),
        (c * hw, 1, w * c, c),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    out_left = torch.empty_strided(
        (n, c, h, w),
        (c * hw, 1, w * c, c),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    out_right_sum = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    out_left_sum = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    partial = torch.empty((2, c, num_k_blocks), device=arg0.device, dtype=torch.float32)

    _dual_where_partial_kernel[(triton.cdiv(c, BLOCK_C), num_k_blocks)](
        arg0,
        arg1,
        arg2,
        arg3,
        out_right,
        out_left,
        partial,
        C=c,
        H=h,
        W=w,
        HW=hw,
        K_TOTAL=k_total,
        NUM_K_BLOCKS=num_k_blocks,
        INPUT_STRIDE_N=arg0.stride(0),
        INPUT_STRIDE_H=arg0.stride(2),
        INPUT_STRIDE_W=arg0.stride(3),
        MASK_STRIDE_N=arg1.stride(0),
        MASK_STRIDE_H=arg1.stride(2),
        MASK_STRIDE_W=arg1.stride(3),
        RIGHT_C_OFFSET=128,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _dual_where_final_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        partial,
        out_right_sum,
        out_left_sum,
        C=c,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_BLOCKS=block_blocks,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    return out_right, out_right_sum, out_left, out_left_sum
