"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete bf16 GPT-J RoPE pointwise scope, including the positional i64 side output, both rotary branches over arg1/arg2, bf16 multiply/add rounding before the final f32 casts, tail preservation, the two non-contiguous layout views, and the all-false adjacent-position bool output, whereas Inductor currently schedules the shared table gather, duplicated rotate-half subgraphs, concat/permute materializations, and side-output pointwise work as generic regions; Inductor cannot do this today because scheduler fusion does not form one producer/consumer group across the shared RoPE coefficients and sibling layout stores while preserving returned view metadata; the fix is SCHEDULER_FUSION: recognize this GPT-J RoPE gather-and-layout pattern and fuse both branches plus side outputs into one direct store kernel."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
ROTARY_PAIRS = ROTARY_DIM // 2
TABLE_DIM = 64
NUMEL = SEQ * HEADS * HEAD_DIM


@triton.jit
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _gptj_rope_pair_kernel(
    table_ptr,
    arg1_ptr,
    arg2_ptr,
    pos_ptr,
    out0_ptr,
    out1_ptr,
    ne_ptr,
    N: tl.constexpr,
    SEQ_: tl.constexpr,
    HEADS_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    ROTARY_DIM_: tl.constexpr,
    ROTARY_PAIRS_: tl.constexpr,
    TABLE_DIM_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    dim = offsets % HEAD_DIM_
    row = offsets // HEAD_DIM_
    seq = row // HEADS_
    head = row - seq * HEADS_
    base = seq * (HEADS_ * HEAD_DIM_) + head * HEAD_DIM_

    rotary = dim < ROTARY_DIM_
    rotary_mask = mask & rotary
    pair = dim // 2
    odd = (dim & 1) == 1
    paired_dim = tl.where(odd, dim - 1, dim + 1)
    rotate_sign = tl.where(odd, 1.0, -1.0)

    coeff_rotate = tl.load(
        table_ptr + seq * TABLE_DIM_ + pair,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    coeff_direct = tl.load(
        table_ptr + seq * TABLE_DIM_ + ROTARY_PAIRS_ + pair,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)

    paired_offsets = base + paired_dim
    x0 = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x0_pair = tl.load(arg1_ptr + paired_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    x1 = tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1_pair = tl.load(arg2_ptr + paired_offsets, mask=rotary_mask, other=0.0).to(tl.float32)

    y0_direct = _round_bf16_to_fp32(x0 * coeff_direct)
    y0_rotate = _round_bf16_to_fp32((x0_pair * rotate_sign) * coeff_rotate)
    y1_direct = _round_bf16_to_fp32(x1 * coeff_direct)
    y1_rotate = _round_bf16_to_fp32((x1_pair * rotate_sign) * coeff_rotate)
    y0_rotary = _round_bf16_to_fp32(y0_direct + y0_rotate)
    y1_rotary = _round_bf16_to_fp32(y1_direct + y1_rotate)

    y0 = tl.where(rotary, y0_rotary, x0)
    y1 = tl.where(rotary, y1_rotary, x1)
    tl.store(out0_ptr + offsets, y0, mask=mask)
    tl.store(out1_ptr + offsets, y1, mask=mask)

    side_mask = offsets < SEQ_
    tl.store(pos_ptr + offsets, offsets.to(tl.int64), mask=side_mask)
    tl.store(ne_ptr + offsets, offsets != offsets, mask=side_mask)


# 42922299: GPT-J bf16 RoPE pair, f32 layout views, i64 positions, bool adjacency side output.
@oracle_impl(hardware="B200", point="42922299", BLOCK=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0_1, arg1_1, arg2_1 = inputs[:3]
    pos_base = torch.empty((SEQ,), device=arg0_1.device, dtype=torch.int64)
    unsqueeze = pos_base.unsqueeze(0)
    out0_base = torch.empty_strided(
        (1, HEADS, SEQ, HEAD_DIM),
        (NUMEL, HEAD_DIM, HEADS * HEAD_DIM, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out1_base = torch.empty_strided(
        (1, HEADS, HEAD_DIM, SEQ),
        (NUMEL, HEAD_DIM, 1, HEADS * HEAD_DIM),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    view_5 = out0_base.as_strided((HEADS, SEQ, HEAD_DIM), (HEAD_DIM, HEADS * HEAD_DIM, 1))
    view_11 = out1_base.as_strided((HEADS, HEAD_DIM, SEQ), (HEAD_DIM, 1, HEADS * HEAD_DIM))
    ne = torch.empty_strided((1, SEQ), (SEQ, 1), device=arg0_1.device, dtype=torch.bool)

    _gptj_rope_pair_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        unsqueeze,
        out0_base,
        out1_base,
        ne,
        N=NUMEL,
        SEQ_=SEQ,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        ROTARY_DIM_=ROTARY_DIM,
        ROTARY_PAIRS_=ROTARY_PAIRS,
        TABLE_DIM_=TABLE_DIM,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return unsqueeze, view_5, view_11, ne
