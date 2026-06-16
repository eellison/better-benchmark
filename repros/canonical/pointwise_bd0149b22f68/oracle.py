"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the full bf16 GPT-J indexed RoPE scope, including the f32 table gather rounded to bf16, returned split coefficient views, both duplicated rotate-half branches, tail preservation, and the two aliased final layout views, whereas Inductor currently emits a separate gather/cast kernel plus a generic pointwise/layout kernel; Inductor cannot do this today because scheduler fusion does not inline the indexed coefficient gather through the shared split/expand/clone views and sibling cat/permute materializations while preserving the returned aliasing metadata; the fix is SCHEDULER_FUSION: teach Inductor to recognize this GPT-J RoPE gather-and-layout pattern and fuse the coefficient gather with both direct output stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
TABLE_DIM = 64
NUMEL = SEQ * HEADS * HEAD_DIM
COEFF_NUMEL = SEQ * TABLE_DIM
_USE_INDUCTOR_NUMERICS = False


@triton.jit
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _gptj_indexed_rope_layout_kernel(
    arg0_ptr,
    arg1_ptr,
    table_ptr,
    index_ptr,
    coeff_ptr,
    out0_ptr,
    out1_ptr,
    N: tl.constexpr,
    COEFF_N: tl.constexpr,
    HEADS_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    ROTARY_DIM_: tl.constexpr,
    TABLE_DIM_: tl.constexpr,
    ROUND_PRODUCTS: tl.constexpr,
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
    pair = dim // 2
    odd = (dim & 1) == 1
    paired_dim = tl.where(odd, dim - 1, dim + 1)
    rotate_sign = tl.where(odd, 1.0, -1.0)
    rotary_mask = mask & rotary

    rotate_index = tl.load(
        index_ptr + seq * TABLE_DIM_ + pair,
        mask=rotary_mask,
        other=0,
    )
    direct_index = tl.load(
        index_ptr + seq * TABLE_DIM_ + ROTARY_DIM_ // 2 + pair,
        mask=rotary_mask,
        other=0,
    )
    coeff_rotate = tl.load(
        table_ptr + rotate_index * TABLE_DIM_ + pair,
        mask=rotary_mask,
        other=0.0,
    )
    coeff_direct = tl.load(
        table_ptr + direct_index * TABLE_DIM_ + ROTARY_DIM_ // 2 + pair,
        mask=rotary_mask,
        other=0.0,
    )
    coeff_rotate = _round_bf16_to_fp32(coeff_rotate)
    coeff_direct = _round_bf16_to_fp32(coeff_direct)

    x0 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    paired_offsets = base + paired_dim
    x0_pair = tl.load(arg0_ptr + paired_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    x1_pair = tl.load(arg1_ptr + paired_offsets, mask=rotary_mask, other=0.0).to(tl.float32)

    if ROUND_PRODUCTS:
        y0_direct = _round_bf16_to_fp32(x0 * coeff_direct)
        y0_rotate = _round_bf16_to_fp32((x0_pair * rotate_sign) * coeff_rotate)
        y1_direct = _round_bf16_to_fp32(x1 * coeff_direct)
        y1_rotate = _round_bf16_to_fp32((x1_pair * rotate_sign) * coeff_rotate)
        y0_rotary = y0_direct + y0_rotate
        y1_rotary = y1_direct + y1_rotate
    else:
        y0_rotary = x0 * coeff_direct + (x0_pair * rotate_sign) * coeff_rotate
        y1_rotary = x1 * coeff_direct + (x1_pair * rotate_sign) * coeff_rotate
    y0 = tl.where(rotary, y0_rotary, x0)
    y1 = tl.where(rotary, y1_rotary, x1)

    tl.store(out0_ptr + offsets, y0, mask=mask)
    tl.store(out1_ptr + offsets, y1, mask=mask)

    coeff_offsets = offsets
    coeff_mask = coeff_offsets < COEFF_N
    coeff_col = coeff_offsets % TABLE_DIM_
    coeff_index = tl.load(index_ptr + coeff_offsets, mask=coeff_mask, other=0)
    coeff_value = tl.load(
        table_ptr + coeff_index * TABLE_DIM_ + coeff_col,
        mask=coeff_mask,
        other=0.0,
    )
    tl.store(coeff_ptr + coeff_offsets, coeff_value, mask=coeff_mask)


# 8b48ba8a: (T([128,4096], bf16), T([128,4096], bf16), T([2048,64], f32), T([1,128,64], i64), ...)
@oracle_impl(hardware="B200", point="8b48ba8a", BLOCK=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    global _USE_INDUCTOR_NUMERICS
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs[:4]
    round_products = not _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        round_products = False

    coeff = torch.empty_strided(
        (1, SEQ, TABLE_DIM),
        (SEQ * TABLE_DIM, TABLE_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    view_8 = torch.empty_strided(
        (HEADS, SEQ, HEAD_DIM),
        (HEAD_DIM, HEADS * HEAD_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    view_9 = torch.empty_strided(
        (HEADS, HEAD_DIM, SEQ),
        (HEAD_DIM, 1, HEADS * HEAD_DIM),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(NUMEL, BLOCK),)
    _gptj_indexed_rope_layout_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        coeff,
        view_8,
        view_9,
        N=NUMEL,
        COEFF_N=COEFF_NUMEL,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        ROTARY_DIM_=ROTARY_DIM,
        TABLE_DIM_=TABLE_DIM,
        ROUND_PRODUCTS=round_products,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )

    unsqueeze_1 = coeff[:, :, : ROTARY_DIM // 2].unsqueeze(2).unsqueeze(4)
    unsqueeze_3 = coeff[:, :, ROTARY_DIM // 2 :].unsqueeze(2).unsqueeze(4)
    return (
        unsqueeze_1,
        unsqueeze_3,
        view_8,
        view_9,
        view_8.permute(0, 2, 1),
        view_9.permute(0, 2, 1),
    )
