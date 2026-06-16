"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 GPT-J interleaved RoPE layout scope in one Triton materialization, including the two returned zero slice_scatter bases, duplicated rotate-half branches with bf16 product and add rounding, tail preservation, and the two returned contiguous bases plus transposed alias views, whereas Inductor lowers the view/permute/expand/clone/slice_scatter graph through generic pointwise-layout scheduling; Inductor cannot do this today because scheduler fusion does not recognize the repeated RoPE rotate-half plus slice_scatter assembly as direct stores into the final returned layouts while preserving visible side outputs and aliases; the fix is SCHEDULER_FUSION: teach pointwise/layout scheduling to fuse GPT-J RoPE slice_scatter patterns into multi-output direct materialization with exact bf16 rounding boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
ROTARY_PAIRS = ROTARY_DIM // 2
HIDDEN = HEADS * HEAD_DIM
NUMEL = SEQ * HIDDEN
ZERO64_NUMEL = SEQ * HEADS * ROTARY_DIM


@triton.jit
def _round_bf16_to_fp32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _gptj_interleaved_rope_kernel(
    arg0_ptr,
    arg1_ptr,
    rotate_coeff_ptr,
    direct_coeff_ptr,
    zero64_ptr,
    zero256_ptr,
    out0_ptr,
    out1_ptr,
    n_elements: tl.constexpr,
    zero64_elements: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements

    dim = offsets % head_dim
    row = offsets // head_dim
    seq = row // heads
    head = row - seq * heads

    rotary = dim < rotary_dim
    rotary_mask = mask & rotary
    pair = dim // 2
    odd = (dim & 1) == 1
    paired_dim = tl.where(odd, dim - 1, dim + 1)
    rotate_sign = tl.where(odd, -1.0, 1.0)

    coeff_offsets = seq * 64 + pair
    rotate_coeff = tl.load(
        rotate_coeff_ptr + coeff_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    direct_coeff = tl.load(
        direct_coeff_ptr + coeff_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)

    arg0_offsets = head * 32768 + dim * 128 + seq
    arg0_pair_offsets = head * 32768 + paired_dim * 128 + seq
    arg1_offsets = head * 32768 + seq * head_dim + dim
    arg1_pair_offsets = head * 32768 + seq * head_dim + paired_dim

    x0 = tl.load(arg0_ptr + arg0_offsets, mask=mask, other=0.0).to(tl.float32)
    x0_pair = tl.load(
        arg0_ptr + arg0_pair_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    x1 = tl.load(arg1_ptr + arg1_offsets, mask=mask, other=0.0).to(tl.float32)
    x1_pair = tl.load(
        arg1_ptr + arg1_pair_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)

    x0_rotate_product = _round_bf16_to_fp32(_f32_mul(x0_pair, rotate_coeff))
    x1_rotate_product = _round_bf16_to_fp32(_f32_mul(x1_pair, rotate_coeff))
    x0_rotate = tl.where(odd, -x0_rotate_product, x0_rotate_product)
    x1_rotate = tl.where(odd, -x1_rotate_product, x1_rotate_product)
    x0_direct = _round_bf16_to_fp32(_f32_mul(x0, direct_coeff))
    x1_direct = _round_bf16_to_fp32(_f32_mul(x1, direct_coeff))
    y0_rotary = _round_bf16_to_fp32(_f32_add(x0_rotate, x0_direct))
    y1_rotary = _round_bf16_to_fp32(_f32_add(x1_rotate, x1_direct))

    y0 = tl.where(rotary, y0_rotary, x0)
    y1 = tl.where(rotary, y1_rotary, x1)
    tl.store(out0_ptr + offsets, y0, mask=mask)
    tl.store(out1_ptr + offsets, y1, mask=mask)
    tl.store(zero256_ptr + offsets, tl.zeros((block,), tl.float32), mask=mask)
    tl.store(
        zero64_ptr + offsets,
        tl.zeros((block,), tl.float32),
        mask=offsets < zero64_elements,
    )


# 40872e86: GPT-J bf16 interleaved RoPE, six outputs with two aliasing view pairs.
@oracle_impl(hardware="B200", point="40872e86", BLOCK=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs[:4]

    zero64 = torch.empty_strided(
        (1, SEQ, HEADS, ROTARY_DIM),
        (SEQ * HEADS * ROTARY_DIM, HEADS * ROTARY_DIM, ROTARY_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    zero256 = torch.empty_strided(
        (1, SEQ, HEADS, HEAD_DIM),
        (NUMEL, HIDDEN, HEAD_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out0 = torch.empty_strided(
        (SEQ, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out1 = torch.empty_strided(
        (SEQ, HIDDEN),
        (HIDDEN, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _gptj_interleaved_rope_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        zero64,
        zero256,
        out0,
        out1,
        n_elements=NUMEL,
        zero64_elements=ZERO64_NUMEL,
        heads=HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        hidden=HIDDEN,
        block=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return zero64, zero256, out0, out0.permute(1, 0), out1, out1.permute(1, 0)
