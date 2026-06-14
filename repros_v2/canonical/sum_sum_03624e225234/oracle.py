"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete captured SqueezeNet bf16 scope by co-materializing both channels-last `where(mask, fill, slice)` outputs and co-reducing their channel sums from the same `[N,H,W]` traversal, preserving the returned bf16 producer tensors while matching Inductor's f32 reduction epilogue for the converted sum outputs. Inductor schedules the two masked slice branches and sibling reductions as generic regions around separate materialized tensors; it cannot do this today because reduction scheduling does not form one full-scope multi-output template with independent accumulators for compatible sibling reductions that share axes and layout while also returning the producer tensors. The fix is SCHEDULER_FUSION: teach the scheduler to fuse such sibling masked reductions with their required producer materialization and dtype epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 32
C = 64
INPUT_C = 128
H = 55
W = 55
HW = H * W
OUTPUT_N_STRIDE = C * HW
REDUCE_ELEMS = N * HW
BLOCK_M = 256
BLOCK_C = 16
NUM_M_BLOCKS = triton.cdiv(REDUCE_ELEMS, BLOCK_M)
FINAL_BLOCK = 512


@triton.jit
def _partial_where_sum_kernel(
    input_ptr,
    right_mask_ptr,
    fill_ptr,
    left_mask_ptr,
    right_out_ptr,
    left_out_ptr,
    partial_ptr,
    BLOCK_M_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
    NUM_M_BLOCKS_: tl.constexpr,
    C_: tl.constexpr,
    INPUT_C_: tl.constexpr,
    HW_: tl.constexpr,
    REDUCE_ELEMS_: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    block_id = tl.program_id(1)
    r = block_id * BLOCK_M_ + tl.arange(0, BLOCK_M_)
    active = (r[:, None] < REDUCE_ELEMS_) & (c_offsets[None, :] < C_)

    n = r // HW_
    spatial = r - n * HW_
    input_base = n[:, None] * (INPUT_C_ * HW_) + spatial[:, None] * INPUT_C_ + c_offsets[None, :]
    output_base = n[:, None] * (C_ * HW_) + spatial[:, None] * C_ + c_offsets[None, :]

    fill = tl.load(fill_ptr)
    right_values = tl.load(input_ptr + input_base + C_, mask=active, other=0.0)
    left_values = tl.load(input_ptr + input_base, mask=active, other=0.0)
    right_mask = tl.load(right_mask_ptr + output_base, mask=active, other=0)
    left_mask = tl.load(left_mask_ptr + output_base, mask=active, other=0)

    right_where = tl.where(right_mask, fill, right_values)
    left_where = tl.where(left_mask, fill, left_values)
    tl.store(right_out_ptr + output_base, right_where, mask=active)
    tl.store(left_out_ptr + output_base, left_where, mask=active)

    right_terms = tl.where(active, right_where.to(tl.float32), 0.0)
    left_terms = tl.where(active, left_where.to(tl.float32), 0.0)
    partial_offset = c_offsets * NUM_M_BLOCKS_ + block_id
    c_mask = c_offsets < C_
    tl.store(partial_ptr + partial_offset, tl.sum(right_terms, axis=0), mask=c_mask)
    tl.store(
        partial_ptr + C_ * NUM_M_BLOCKS_ + partial_offset,
        tl.sum(left_terms, axis=0),
        mask=c_mask,
    )


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    right_sum_ptr,
    left_sum_ptr,
    NUM_M_BLOCKS_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    blocks = tl.arange(0, BLOCK_B)
    valid = (blocks[:, None] < NUM_M_BLOCKS_) & (c_offsets[None, :] < C_)
    partial_offset = c_offsets[None, :] * NUM_M_BLOCKS_ + blocks[:, None]

    right_partials = tl.load(partial_ptr + partial_offset, mask=valid, other=0.0)
    left_partials = tl.load(
        partial_ptr + C_ * NUM_M_BLOCKS_ + partial_offset,
        mask=valid,
        other=0.0,
    )
    c_mask = c_offsets < C_
    tl.store(
        right_sum_ptr + c_offsets,
        tl.sum(right_partials.to(tl.float32), axis=0),
        mask=c_mask,
    )
    tl.store(
        left_sum_ptr + c_offsets,
        tl.sum(left_partials.to(tl.float32), axis=0),
        mask=c_mask,
    )


# squeezenet1_1 train/infer, channels-last N=32 C=64 H=W=55, input halves 0:64 and 64:128.
@oracle_impl(
    hardware="B200",
    point="185fdb55",
    BLOCK_M_=BLOCK_M,
    BLOCK_C_=BLOCK_C,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK_M_: int, BLOCK_C_: int, num_warps: int):
    input_tensor, right_mask, fill, left_mask = inputs

    right_out = torch.empty_strided(
        (N, C, H, W),
        (OUTPUT_N_STRIDE, 1, W * C, C),
        device=input_tensor.device,
        dtype=torch.bfloat16,
    )
    right_sum = torch.empty_strided((C,), (1,), device=input_tensor.device, dtype=torch.float32)
    left_out = torch.empty_strided(
        (N, C, H, W),
        (OUTPUT_N_STRIDE, 1, W * C, C),
        device=input_tensor.device,
        dtype=torch.bfloat16,
    )
    left_sum = torch.empty_strided((C,), (1,), device=input_tensor.device, dtype=torch.float32)
    partial = torch.empty(
        (2, C, NUM_M_BLOCKS),
        device=input_tensor.device,
        dtype=torch.float32,
    )

    _partial_where_sum_kernel[(triton.cdiv(C, BLOCK_C_), NUM_M_BLOCKS)](
        input_tensor,
        right_mask,
        fill,
        left_mask,
        right_out,
        left_out,
        partial,
        BLOCK_M_=BLOCK_M_,
        BLOCK_C_=BLOCK_C_,
        NUM_M_BLOCKS_=NUM_M_BLOCKS,
        C_=C,
        INPUT_C_=INPUT_C,
        HW_=HW,
        REDUCE_ELEMS_=REDUCE_ELEMS,
        num_warps=num_warps,
    )
    _finalize_sum_kernel[(triton.cdiv(C, BLOCK_C_),)](
        partial,
        right_sum,
        left_sum,
        NUM_M_BLOCKS_=NUM_M_BLOCKS,
        C_=C,
        BLOCK_C_=BLOCK_C_,
        BLOCK_B=FINAL_BLOCK,
        num_warps=8,
    )
    return right_out, right_sum, left_out, left_sum
