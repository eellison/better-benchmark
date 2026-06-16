"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Mistral bf16 rotary-position pointwise scope, including the f32 iota/inverse-frequency phase table, natural cos/sin with bf16 casts into the returned `[1,1000,128]` tables, rotary application for query and key in the returned permute-layout strides, and the grouped-key expand/clone/view materialization directly in contiguous repeated-head layout; Inductor currently lowers the generated trigonometric table, rotate-half slice/neg/cat/mul/add branches, and grouped-key repeat clone as separate generic pointwise/layout regions; Inductor cannot do this today because its scheduler cannot keep the generated RoPE table shared across sibling Q/K rotary consumers while assigning one key producer tile to both the non-contiguous key output and the repeated-head side output; the fix is SCHEDULER_FUSION: teach pointwise/layout scheduling to fuse generated RoPE tables through rotary-style split-half consumers and grouped-key expand/clone users with multi-output layout ownership."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEQ = 1000
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
HALF = 64
GROUPS = Q_HEADS // KV_HEADS
TABLE_NUMEL = SEQ * HEAD_DIM
PAIR_NUMEL = SEQ * Q_HEADS * HALF


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
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rope_table_kernel(
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    N: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    dim = offsets % D
    seq = offsets // D
    freq_dim = dim % HALF_D
    inv_freq = tl.load(inv_freq_ptr + freq_dim, mask=mask, other=0.0).to(tl.float32)
    phase = _f32_mul(inv_freq, seq.to(tl.float32))
    cos_value = _f32_mul(libdevice.cos(phase), 1.0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    sin_value = _f32_mul(libdevice.sin(phase), 1.0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(cos_ptr + offsets, cos_value, mask=mask)
    tl.store(sin_ptr + offsets, sin_value, mask=mask)


@triton.jit
def _rotary_q_k_repeat_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    out_repeat_ptr,
    N_PAIRS: tl.constexpr,
    S: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    GROUPS_C: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    pair_offsets = tl.program_id(0) * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    mask = pair_offsets < N_PAIRS

    half_d = pair_offsets % HALF_D
    q_head = (pair_offsets // HALF_D) % QH
    seq = (pair_offsets // (HALF_D * QH)) % S

    coeff_base = seq * D + half_d
    cos_lo = tl.load(cos_ptr + coeff_base, mask=mask, other=0.0).to(tl.float32)
    cos_hi = tl.load(cos_ptr + coeff_base + HALF_D, mask=mask, other=0.0).to(
        tl.float32
    )
    sin_lo = tl.load(sin_ptr + coeff_base, mask=mask, other=0.0).to(tl.float32)
    sin_hi = tl.load(sin_ptr + coeff_base + HALF_D, mask=mask, other=0.0).to(
        tl.float32
    )

    q_base = (seq * QH + q_head) * D + half_d
    q_lo = tl.load(q_ptr + q_base, mask=mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptr + q_base + HALF_D, mask=mask, other=0.0).to(tl.float32)
    q_mul_lo = _round_to_bf16_f32(_f32_mul(q_lo, cos_lo))
    q_rot_lo = _round_to_bf16_f32(_f32_mul(-q_hi, sin_lo))
    q_mul_hi = _round_to_bf16_f32(_f32_mul(q_hi, cos_hi))
    q_rot_hi = _round_to_bf16_f32(_f32_mul(q_lo, sin_hi))
    q_out_lo = q_mul_lo + q_rot_lo
    q_out_hi = q_mul_hi + q_rot_hi

    tl.store(
        out_q_ptr + q_base,
        q_out_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )
    tl.store(
        out_q_ptr + q_base + HALF_D,
        q_out_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )

    k_mask = mask & (q_head < KVH)
    k_base = (seq * KVH + q_head) * D + half_d
    k_lo = tl.load(k_ptr + k_base, mask=k_mask, other=0.0).to(tl.float32)
    k_hi = tl.load(k_ptr + k_base + HALF_D, mask=k_mask, other=0.0).to(tl.float32)
    k_mul_lo = _round_to_bf16_f32(_f32_mul(k_lo, cos_lo))
    k_rot_lo = _round_to_bf16_f32(_f32_mul(-k_hi, sin_lo))
    k_mul_hi = _round_to_bf16_f32(_f32_mul(k_hi, cos_hi))
    k_rot_hi = _round_to_bf16_f32(_f32_mul(k_lo, sin_hi))
    k_out_lo = k_mul_lo + k_rot_lo
    k_out_hi = k_mul_hi + k_rot_hi

    tl.store(
        out_k_ptr + k_base,
        k_out_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=k_mask,
    )
    tl.store(
        out_k_ptr + k_base + HALF_D,
        k_out_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=k_mask,
    )

    for group in tl.static_range(0, GROUPS_C):
        out_head = q_head * GROUPS_C + group
        repeat_base = (out_head * S + seq) * D + half_d
        tl.store(
            out_repeat_ptr + repeat_base,
            k_out_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=k_mask,
        )
        tl.store(
            out_repeat_ptr + repeat_base + HALF_D,
            k_out_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=k_mask,
        )


@oracle_impl(
    hardware="B200",
    point="cc5826e9",
    TABLE_BLOCK=256,
    BLOCK_PAIRS=1024,
    num_warps=4,
)
def oracle_forward(inputs, *, TABLE_BLOCK: int, BLOCK_PAIRS: int, num_warps: int):
    q_mm, inv_freq, k_mm, *_shape_params = inputs
    cos_out = torch.empty_strided(
        (1, SEQ, HEAD_DIM),
        (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=inv_freq.device,
        dtype=torch.bfloat16,
    )
    sin_out = torch.empty_strided(
        (1, SEQ, HEAD_DIM),
        (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=inv_freq.device,
        dtype=torch.bfloat16,
    )
    q_out = torch.empty_strided(
        (1, Q_HEADS, SEQ, HEAD_DIM),
        (SEQ * Q_HEADS * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )
    k_out = torch.empty_strided(
        (1, KV_HEADS, SEQ, HEAD_DIM),
        (SEQ * KV_HEADS * HEAD_DIM, HEAD_DIM, KV_HEADS * HEAD_DIM, 1),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )
    repeat_out = torch.empty_strided(
        (1, Q_HEADS, SEQ, HEAD_DIM),
        (Q_HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )

    _rope_table_kernel[(triton.cdiv(TABLE_NUMEL, TABLE_BLOCK),)](
        inv_freq,
        cos_out,
        sin_out,
        N=TABLE_NUMEL,
        S=SEQ,
        D=HEAD_DIM,
        HALF_D=HALF,
        BLOCK=TABLE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _rotary_q_k_repeat_kernel[(triton.cdiv(PAIR_NUMEL, BLOCK_PAIRS),)](
        q_mm,
        k_mm,
        cos_out,
        sin_out,
        q_out,
        k_out,
        repeat_out,
        N_PAIRS=PAIR_NUMEL,
        S=SEQ,
        QH=Q_HEADS,
        KVH=KV_HEADS,
        D=HEAD_DIM,
        HALF_D=HALF,
        GROUPS_C=GROUPS,
        BLOCK_PAIRS=BLOCK_PAIRS,
        num_warps=num_warps,
        num_stages=3,
    )
    return cos_out, sin_out, q_out, k_out, repeat_out
