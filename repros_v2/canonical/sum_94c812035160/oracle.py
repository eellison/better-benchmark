"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes
the complete Demucs bf16 GLU-derivative cat plus channel-sum scope, including
f32 sigmoid/multiply arithmetic, the visible contiguous bf16 `[4,128,92842]`
cat output, and the returned f32 copy of the bf16 `sum([0, 2])`. Inductor
currently lowers the slice/sigmoid/two-branch multiply/cat materialization and
the sibling reduction over that materialized bf16 tensor as generic producer and
reduction work. It cannot do this today because scheduler/codegen does not form
one split-K materialize-plus-reduce plan that shares the sigmoid producer across
both cat halves while preserving the explicit bf16 cast before the reduction;
the fix is COOPERATIVE_SPLIT_K: emit bf16 stores and per-channel partial sums
from the same producer traversal, then finalize the rounded channel sums once."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 64
C_TOTAL = 128
TIME = 92842
BLOCK_C = 16
BLOCK_T = 256
TILES_T = triton.cdiv(TIME, BLOCK_T)
NUM_PARTIALS = BATCH * TILES_T
FINAL_BLOCK = triton.next_power_of_2(NUM_PARTIALS)


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _produce_partials_kernel(
    grad_ptr,
    gate_pair_ptr,
    out_ptr,
    partials_ptr,
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
    valid = (channels[:, None] < 64) & (times[None, :] < 92842)

    half_offsets = batch * (64 * 92842) + c * 92842 + t
    full_offsets = batch * (128 * 92842) + c * 92842 + t

    grad = tl.load(grad_ptr + half_offsets, mask=valid, other=0.0).to(tl.float32)
    first_half = tl.load(gate_pair_ptr + full_offsets, mask=valid, other=0.0).to(tl.float32)
    gate = tl.load(
        gate_pair_ptr + full_offsets + 64 * 92842,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    sigmoid = _f32_div(1.0, _f32_add(libdevice.exp(_f32_sub(0.0, gate)), 1.0))
    first = _f32_mul(sigmoid, grad)
    second = _f32_mul(
        _f32_mul(_f32_mul(_f32_sub(1.0, sigmoid), sigmoid), first_half),
        grad,
    )
    first_bf16 = first.to(tl.bfloat16, fp_downcast_rounding="rtne")
    second_bf16 = second.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + full_offsets, first_bf16, mask=valid)
    tl.store(out_ptr + full_offsets + 64 * 92842, second_bf16, mask=valid)

    partial_offsets = (batch * 363 + t_tile) * 64 + channels
    channel_mask = channels < 64
    tl.store(
        partials_ptr + partial_offsets,
        tl.sum(tl.where(valid, first_bf16.to(tl.float32), 0.0), axis=1),
        mask=channel_mask,
    )
    tl.store(
        partials_ptr + 363 * 4 * 64 + partial_offsets,
        tl.sum(tl.where(valid, second_bf16.to(tl.float32), 0.0), axis=1),
        mask=channel_mask,
    )


@triton.jit
def _finalize_sum_kernel(
    partials_ptr,
    sum_ptr,
    BLOCK_C_: tl.constexpr,
    BLOCK_P_: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    partials = tl.arange(0, BLOCK_P_)
    mask = (partials[:, None] < 1452) & (channels[None, :] < 64)
    offsets = partials[:, None] * 64 + channels[None, :]

    first = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(
        partials_ptr + 1452 * 64 + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    channel_mask = channels < 64
    first_sum = tl.sum(first, axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    second_sum = tl.sum(second, axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + channels, first_sum, mask=channel_mask)
    tl.store(sum_ptr + 64 + channels, second_sum, mask=channel_mask)


@oracle_impl(
    hardware="B200",
    point="be14a97f",
    BLOCK_C=BLOCK_C,
    BLOCK_T=BLOCK_T,
    FINAL_BLOCK=FINAL_BLOCK,
    num_warps_produce=8,
    num_warps_final=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_T: int,
    FINAL_BLOCK: int,
    num_warps_produce: int,
    num_warps_final: int,
):
    grad, gate_pair = inputs
    device = grad.device
    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty_strided(
        (2, NUM_PARTIALS, C_HALF),
        (NUM_PARTIALS * C_HALF, C_HALF, 1),
        device=device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((C_TOTAL,), (1,), device=device, dtype=torch.float32)

    _produce_partials_kernel[(BATCH, triton.cdiv(C_HALF, BLOCK_C), TILES_T)](
        grad,
        gate_pair,
        out,
        partials,
        BLOCK_C_=BLOCK_C,
        BLOCK_T_=BLOCK_T,
        num_warps=num_warps_produce,
        num_stages=3,
    )
    _finalize_sum_kernel[(triton.cdiv(C_HALF, BLOCK_C),)](
        partials,
        sum_out,
        BLOCK_C_=BLOCK_C,
        BLOCK_P_=FINAL_BLOCK,
        num_warps=num_warps_final,
        num_stages=3,
    )
    return out, sum_out
