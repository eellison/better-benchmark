"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-J bf16 RoPE pointwise/layout scope for both branches, including expanded pair coefficients, slice_scatter rotate-half bases, arg5 tail/base assembly, and the returned contiguous plus transposed views; whereas Inductor lowers the decomposed permute/expand/clone/mul/select/slice_scatter/add/view/permute graph as generic pointwise-layout work; Inductor cannot do this today because scheduler fusion does not sink the duplicated RoPE scatter assembly through the final view/permute output contract; the fix is SCHEDULER_FUSION: recognize GPT-J pair-RoPE scatter assembly and fuse both branches directly into the final backing strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
HIDDEN = HEADS * HEAD_DIM
ROWS = SEQ * HEADS


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _bf16_add(a, b):
    return _bf16(_f32_add(a, b))


@triton.jit
def _bf16_mul(a, b):
    return _bf16(_f32_mul(a, b))


@triton.jit
def _gptj_rope_bf16_kernel(
    q_in_ptr,
    k_in_ptr,
    coeff_cos_ptr,
    scatter_base_ptr,
    coeff_sin_ptr,
    full_base_ptr,
    q_out_ptr,
    k_out_ptr,
    SEQ_: tl.constexpr,
    ROWS_: tl.constexpr,
    HEADS_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    ROTARY_DIM_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    row_mask = rows < ROWS_
    elem_mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM_)

    seq = rows // HEADS_
    head = rows - seq * HEADS_
    pair = dims // 2
    paired_dim = dims ^ 1
    rotary = dims < ROTARY_DIM_
    rotary_mask = row_mask[:, None] & rotary[None, :]
    is_even = (dims & 1) == 0

    out_offsets = seq[:, None] * HIDDEN_ + head[:, None] * HEAD_DIM_ + dims[None, :]
    scatter_offsets = seq[:, None] * (HEADS_ * ROTARY_DIM_) + head[:, None] * ROTARY_DIM_ + dims[None, :]
    q_offsets = head[:, None] * (HEAD_DIM_ * SEQ_) + dims[None, :] * SEQ_ + seq[:, None]
    q_pair_offsets = head[:, None] * (HEAD_DIM_ * SEQ_) + paired_dim[None, :] * SEQ_ + seq[:, None]
    k_offsets = head[:, None] * (SEQ_ * HEAD_DIM_) + seq[:, None] * HEAD_DIM_ + dims[None, :]
    k_pair_offsets = head[:, None] * (SEQ_ * HEAD_DIM_) + seq[:, None] * HEAD_DIM_ + paired_dim[None, :]
    coeff_offsets = seq[:, None] * ROTARY_DIM_ + pair[None, :]

    cosv = tl.load(coeff_cos_ptr + coeff_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    sinv = tl.load(coeff_sin_ptr + coeff_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    scatter_base = tl.load(scatter_base_ptr + scatter_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    full_base = tl.load(full_base_ptr + out_offsets, mask=elem_mask, other=0.0).to(tl.float32)

    q = _bf16(tl.load(q_in_ptr + q_offsets, mask=elem_mask, other=0.0).to(tl.float32))
    q_pair = _bf16(tl.load(q_in_ptr + q_pair_offsets, mask=rotary_mask, other=0.0).to(tl.float32))
    q_pair_cos = _bf16_mul(q_pair, cosv)
    q_scatter_even = _bf16_add(scatter_base, q_pair_cos)
    q_scatter_odd = _bf16_add(-q_pair_cos, scatter_base)
    q_scatter = tl.where(is_even[None, :], q_scatter_even, q_scatter_odd)
    q_direct = _bf16_mul(q, sinv)
    q_rotary = _bf16_add(full_base, _bf16_add(q_scatter, q_direct))
    q_tail = _bf16_add(q, full_base)
    q_out = tl.where(rotary[None, :], q_rotary, q_tail)
    tl.store(q_out_ptr + out_offsets, q_out, mask=elem_mask)

    k = tl.load(k_in_ptr + k_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    k_pair = tl.load(k_in_ptr + k_pair_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    k_pair_cos = _bf16_mul(k_pair, cosv)
    k_scatter_even = _bf16_add(scatter_base, k_pair_cos)
    k_scatter_odd = _bf16_add(-k_pair_cos, scatter_base)
    k_scatter = tl.where(is_even[None, :], k_scatter_even, k_scatter_odd)
    k_direct = _bf16_mul(k, sinv)
    k_rotary = _bf16_add(full_base, _bf16_add(k_scatter, k_direct))
    k_tail = _bf16_add(k, full_base)
    k_out = tl.where(rotary[None, :], k_rotary, k_tail)
    tl.store(k_out_ptr + out_offsets, k_out, mask=elem_mask)


@oracle_impl(hardware="B200", point="0231a2f7", BLOCK_M=1, BLOCK_D=256, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_D, num_warps):
    q_in, k_in, coeff_cos, scatter_base, coeff_sin, full_base = inputs[:6]
    q_out = torch.empty_strided((SEQ, HIDDEN), (HIDDEN, 1), device=q_in.device, dtype=torch.bfloat16)
    k_out = torch.empty_strided((SEQ, HIDDEN), (HIDDEN, 1), device=q_in.device, dtype=torch.bfloat16)
    _gptj_rope_bf16_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        q_in,
        k_in,
        coeff_cos,
        scatter_base,
        coeff_sin,
        full_base,
        q_out,
        k_out,
        SEQ_=SEQ,
        ROWS_=ROWS,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        ROTARY_DIM_=ROTARY_DIM,
        HIDDEN_=HIDDEN,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=3,
    )
    return q_out, q_out.permute(1, 0), k_out, k_out.permute(1, 0)
