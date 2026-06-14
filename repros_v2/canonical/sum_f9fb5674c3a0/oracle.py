"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Demucs GLU-style producer, returned `[4,512,5979]` bf16 channel-cat tensor in Inductor's fp32-add numeric envelope, and sibling eager-compatible bf16 channel sum in one Triton producer pass plus a tiny finalizer, whereas Inductor schedules the pointwise add/sigmoid/mul/cat materialization and the reduction over that materialized bf16 tensor as generic regions; Inductor cannot do this today because scheduler/codegen does not fuse a live pointwise/cat output with a sibling reduction while preserving both the compiled returned-tensor envelope and the captured bf16 reduction dtype; the fix is SCHEDULER_FUSION: add a multi-output pointwise-plus-reduction schedule that emits the returned layout and partial channel sums from the same producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
HALF_C = 256
OUT_C = 512
T = 5979
BLOCK_C = 16
BLOCK_T = 256
TILES_T = triton.cdiv(T, BLOCK_T)
NUM_PARTIALS = BATCH * TILES_T
FINAL_BLOCK = 128


@triton.jit
def _producer_partial_sum_kernel(
    add_lhs_ptr,
    add_rhs_ptr,
    gate_src_ptr,
    out_ptr,
    partial_ptr,
    HALF_C_: tl.constexpr,
    OUT_C_: tl.constexpr,
    T_: tl.constexpr,
    TILES_T_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
    BLOCK_T_: tl.constexpr,
):
    batch = tl.program_id(0)
    c_block = tl.program_id(1)
    t_tile = tl.program_id(2)
    channels = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    times = t_tile * BLOCK_T_ + tl.arange(0, BLOCK_T_)
    c = channels[:, None]
    t = times[None, :]
    valid = (channels[:, None] < OUT_C_) & (times[None, :] < T_)

    source_channel = channels % HALF_C_
    sc = source_channel[:, None]
    add_offsets = batch * (HALF_C_ * T_) + sc * T_ + t
    x_offsets = batch * (OUT_C_ * T_) + sc * T_ + t
    gate_offsets = batch * (OUT_C_ * T_) + (sc + HALF_C_) * T_ + t
    out_offsets = batch * (OUT_C_ * T_) + c * T_ + t

    lhs = tl.load(add_lhs_ptr + add_offsets, mask=valid, other=0.0).to(tl.float32)
    rhs = tl.load(add_rhs_ptr + add_offsets, mask=valid, other=0.0).to(tl.float32)
    add_compiled = lhs + rhs
    add_eager = add_compiled.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    x_value = tl.load(gate_src_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)
    gate = tl.load(gate_src_ptr + gate_offsets, mask=valid, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(gate)

    first_half = sigmoid * add_compiled
    second_half = (1.0 - sigmoid) * sigmoid * x_value * add_compiled
    produced = tl.where(channels[:, None] < HALF_C_, first_half, second_half)
    produced_bf16 = produced.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + out_offsets, produced_bf16, mask=valid)

    first_half_eager = sigmoid * add_eager
    second_half_eager = (1.0 - sigmoid) * sigmoid * x_value * add_eager
    produced_eager = tl.where(channels[:, None] < HALF_C_, first_half_eager, second_half_eager)
    produced_eager_bf16 = produced_eager.to(tl.bfloat16, fp_downcast_rounding="rtne")
    partial = tl.sum(tl.where(valid, produced_eager_bf16.to(tl.float32), 0.0), axis=1)
    partial_index = (batch * TILES_T_ + t_tile) * OUT_C_ + channels
    tl.store(partial_ptr + partial_index, partial, mask=channels < OUT_C_)


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    sum_ptr,
    OUT_C_: tl.constexpr,
    NUM_PARTIALS_: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    channel = tl.program_id(0)
    partials = tl.arange(0, BLOCK_P)
    mask = partials < NUM_PARTIALS_
    values = tl.load(
        partial_ptr + partials * OUT_C_ + channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(sum_ptr + channel, total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32))


# 0c6e91db: Demucs train bf16 GLU producer plus channel reduction.
@oracle_impl(hardware="B200", point="0c6e91db")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1 = inputs
    device = arg0_1.device
    out = torch.empty_strided(
        (BATCH, OUT_C, T),
        (OUT_C * T, T, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (NUM_PARTIALS, OUT_C),
        (OUT_C, 1),
        device=device,
        dtype=torch.float32,
    )
    out_sum = torch.empty_strided((OUT_C,), (1,), device=device, dtype=torch.float32)

    _producer_partial_sum_kernel[(BATCH, triton.cdiv(OUT_C, BLOCK_C), TILES_T)](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        partial,
        HALF_C_=HALF_C,
        OUT_C_=OUT_C,
        T_=T,
        TILES_T_=TILES_T,
        BLOCK_C_=BLOCK_C,
        BLOCK_T_=BLOCK_T,
        num_warps=8,
        num_stages=3,
    )
    _finalize_sum_kernel[(OUT_C,)](
        partial,
        out_sum,
        OUT_C_=OUT_C,
        NUM_PARTIALS_=NUM_PARTIALS,
        BLOCK_P=FINAL_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return out, out_sum
