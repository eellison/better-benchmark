"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Demucs bf16 GLU-style cat side output in Inductor's compiled f32-add numeric envelope while accumulating the sibling eager-compatible bf16 channel sum from the rounded add path over the same split batch-time producer tiles, whereas Inductor lowers the add/sigmoid/multiply/cat materialization and the `[2048]` reduction as generic producer and reduction schedules over the logical cat domain; Inductor cannot do this today because scheduler/codegen has no multi-output pointwise-plus-reduction template that shares the sigmoid producer across the two cat halves while preserving both the compiled returned-tensor envelope and the captured bf16 reduction dtype; the fix is SCHEDULER_FUSION: emit the visible bf16 cat and per-tile channel partials from one producer traversal, then finalize the rounded channel sum once."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 1024
C_TOTAL = 2048
TIME = 372
K_TOTAL = BATCH * TIME


@triton.jit
def _producer_partial_kernel(
    arg0_ptr,
    arg1_ptr,
    gate_ptr,
    out_ptr,
    partial_ptr,
    C_HALF_: tl.constexpr,
    C_TOTAL_: tl.constexpr,
    TIME_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    k_block = tl.program_id(1)

    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    batch = k // TIME_
    time = k - batch * TIME_
    c_mask = c < C_HALF_
    k_mask = k < K_TOTAL_
    mask = k_mask[:, None] & c_mask[None, :]

    half_offsets = batch[:, None] * C_HALF_ * TIME_ + c[None, :] * TIME_ + time[:, None]
    total_offsets = batch[:, None] * C_TOTAL_ * TIME_ + c[None, :] * TIME_ + time[:, None]

    a = tl.load(arg0_ptr + half_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(arg1_ptr + half_offsets, mask=mask, other=0.0).to(tl.float32)
    add_compiled = a + b
    add_eager = add_compiled.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    first = tl.load(gate_ptr + total_offsets, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(
        gate_ptr + total_offsets + C_HALF_ * TIME_,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sig = tl.sigmoid(second)
    out_first = (sig * add_compiled).to(tl.bfloat16, fp_downcast_rounding="rtne")
    out_second = (((1.0 - sig) * sig * first) * add_compiled).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    sum_first = (sig * add_eager).to(tl.bfloat16, fp_downcast_rounding="rtne")
    sum_second = (((1.0 - sig) * sig * first) * add_eager).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    tl.store(out_ptr + total_offsets, out_first, mask=mask)
    tl.store(out_ptr + total_offsets + C_HALF_ * TIME_, out_second, mask=mask)

    partial_first = tl.sum(tl.where(mask, sum_first.to(tl.float32), 0.0), axis=0)
    partial_second = tl.sum(tl.where(mask, sum_second.to(tl.float32), 0.0), axis=0)
    partial_offsets = k_block * C_HALF_ + c
    tl.store(partial_ptr + partial_offsets, partial_first, mask=c_mask)
    tl.store(partial_ptr + tl.cdiv(K_TOTAL_, BLOCK_K) * C_HALF_ + partial_offsets, partial_second, mask=c_mask)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    C_HALF_: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    c_mask = c < C_HALF_
    mask = (tiles[:, None] < NUM_K_BLOCKS) & c_mask[None, :]
    offsets = tiles[:, None] * C_HALF_ + c[None, :]

    first = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(
        partial_ptr + NUM_K_BLOCKS * C_HALF_ + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sum_first = tl.sum(first, axis=0)
    sum_second = tl.sum(second, axis=0)
    rounded_first = sum_first.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    rounded_second = sum_second.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + c, rounded_first, mask=c_mask)
    tl.store(sum_ptr + C_HALF_ + c, rounded_second, mask=c_mask)


# a7f82378: torchbench Demucs train, bf16 gated cat [4,2048,372] plus bf16 channel sum.
@oracle_impl(
    hardware="B200",
    point="a7f82378",
    BLOCK_K=128,
    BLOCK_C=16,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, num_warps: int):
    arg0_1, arg1_1, arg2_1 = inputs

    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    num_k_blocks = triton.cdiv(K_TOTAL, BLOCK_K)
    partial = torch.empty(
        (2, num_k_blocks, C_HALF),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    reduced = torch.empty((C_TOTAL,), device=arg0_1.device, dtype=torch.float32)

    grid = (triton.cdiv(C_HALF, BLOCK_C), num_k_blocks)
    _producer_partial_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        partial,
        C_HALF_=C_HALF,
        C_TOTAL_=C_TOTAL,
        TIME_=TIME,
        K_TOTAL_=K_TOTAL,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=4,
    )
    _final_sum_kernel[(triton.cdiv(C_HALF, BLOCK_C),)](
        partial,
        reduced,
        C_HALF_=C_HALF,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_TILES=triton.next_power_of_2(num_k_blocks),
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=1,
    )
    return out, reduced
