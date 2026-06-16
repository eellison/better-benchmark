"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the full Gemma rotary embedding pointwise scope into one layout-aware Triton kernel, preserving the returned non-contiguous permuted Q and K rotary outputs while materializing the expand/clone/view repeated-K output directly in its contiguous 8-head layout; Inductor currently decomposes the rotate-half slice/cat, bf16 mul/add, permuted stores, and repeated-K clone materialization because its scheduler cannot assign the different returned layouts and repeat-expansion side write to one producer tile; the fix is SCHEDULER_FUSION: teach pointwise/layout scheduling to fuse rotary-style slice/cat producers with expand/clone consumers and generate multi-output tiles with per-output layout ownership."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _rotary_q_k_repeat_kernel(
    q_ptr,
    cos_ptr,
    sin_ptr,
    k_ptr,
    out_q_ptr,
    out_k_ptr,
    out_rep_ptr,
    N_PAIRS: tl.constexpr,
    S: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    pair_offsets = tl.program_id(0) * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    mask = pair_offsets < N_PAIRS

    half_d = pair_offsets % HALF
    q_head = (pair_offsets // HALF) % QH
    seq = (pair_offsets // (HALF * QH)) % S
    batch = pair_offsets // (HALF * QH * S)

    q_base = ((batch * S + seq) * QH + q_head) * D + half_d
    cos_base = seq * D + half_d

    q_lo = tl.load(q_ptr + q_base, mask=mask).to(tl.float32)
    q_hi = tl.load(q_ptr + q_base + HALF, mask=mask).to(tl.float32)
    cos_lo = tl.load(cos_ptr + cos_base, mask=mask).to(tl.float32)
    cos_hi = tl.load(cos_ptr + cos_base + HALF, mask=mask).to(tl.float32)
    sin_lo = tl.load(sin_ptr + cos_base, mask=mask).to(tl.float32)
    sin_hi = tl.load(sin_ptr + cos_base + HALF, mask=mask).to(tl.float32)

    q_mul_lo = _round_to_bf16_f32(q_lo * cos_lo)
    q_rot_lo = _round_to_bf16_f32((-q_hi) * sin_lo)
    q_mul_hi = _round_to_bf16_f32(q_hi * cos_hi)
    q_rot_hi = _round_to_bf16_f32(q_lo * sin_hi)
    out_q_lo = q_mul_lo + q_rot_lo
    out_q_hi = q_mul_hi + q_rot_hi

    tl.store(out_q_ptr + q_base, out_q_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
    tl.store(out_q_ptr + q_base + HALF, out_q_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)

    k_mask = mask & (q_head < KH)
    k_base = ((batch * S + seq) * KH + q_head) * D + half_d
    k_lo = tl.load(k_ptr + k_base, mask=k_mask).to(tl.float32)
    k_hi = tl.load(k_ptr + k_base + HALF, mask=k_mask).to(tl.float32)

    k_mul_lo = _round_to_bf16_f32(k_lo * cos_lo)
    k_rot_lo = _round_to_bf16_f32((-k_hi) * sin_lo)
    k_mul_hi = _round_to_bf16_f32(k_hi * cos_hi)
    k_rot_hi = _round_to_bf16_f32(k_lo * sin_hi)
    out_k_lo = k_mul_lo + k_rot_lo
    out_k_hi = k_mul_hi + k_rot_hi

    tl.store(out_k_ptr + k_base, out_k_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=k_mask)
    tl.store(out_k_ptr + k_base + HALF, out_k_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=k_mask)

    for group in tl.static_range(0, GROUPS):
        out_head = q_head * GROUPS + group
        rep_base = ((batch * QH + out_head) * S + seq) * D + half_d
        tl.store(out_rep_ptr + rep_base, out_k_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=k_mask)
        tl.store(out_rep_ptr + rep_base + HALF, out_k_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=k_mask)


@oracle_impl(hardware="B200", point="9e133e29", BLOCK_PAIRS=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK_PAIRS, num_warps):
    q_mm, cos, sin, k_mm, *_shape_params = inputs
    batch = int(cos.shape[0])
    seq = int(cos.shape[1])
    head_dim = int(cos.shape[2])
    q_heads = int(q_mm.shape[1]) // head_dim
    k_heads = int(k_mm.shape[1]) // head_dim
    half = head_dim // 2
    groups = q_heads // k_heads
    n_pairs = batch * seq * q_heads * half

    out_q = torch.empty_strided(
        (batch, q_heads, seq, head_dim),
        (seq * q_heads * head_dim, head_dim, q_heads * head_dim, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )
    out_k = torch.empty_strided(
        (batch, k_heads, seq, head_dim),
        (seq * k_heads * head_dim, head_dim, k_heads * head_dim, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )
    out_rep = torch.empty_strided(
        (batch, q_heads, seq, head_dim),
        (q_heads * seq * head_dim, seq * head_dim, head_dim, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )

    _rotary_q_k_repeat_kernel[(triton.cdiv(n_pairs, BLOCK_PAIRS),)](
        q_mm,
        cos,
        sin,
        k_mm,
        out_q,
        out_k,
        out_rep,
        N_PAIRS=n_pairs,
        S=seq,
        QH=q_heads,
        KH=k_heads,
        D=head_dim,
        HALF=half,
        GROUPS=groups,
        BLOCK_PAIRS=BLOCK_PAIRS,
        num_warps=num_warps,
        num_stages=3,
    )
    return out_q, out_k, out_rep
