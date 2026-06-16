"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete Mistral bf16 rotary-position scope for query, key, and repeated key into one layout-aware Triton kernel, preserving the query and key permuted output strides while materializing the key expand/clone/view output directly in contiguous repeated-head layout, whereas Inductor lowers the rotate-half slice/neg/cat/mul/add branches and the grouped-key repeat materialization as generic pointwise and layout work; Inductor cannot do this today because its scheduler cannot assign the sibling rotary producer to two different non-contiguous output layouts plus a repeated-head clone/view side output in one producer tile; the fix is SCHEDULER_FUSION: teach pointwise/layout scheduling to fuse rotary-style slice/cat producers through grouped-key expand/clone consumers and generate multi-output tiles with per-output layout ownership."""

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
    out_repeat_ptr,
    N_PAIRS: tl.constexpr,
    S: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
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
    coeff_base = seq * D + half_d

    q_lo = tl.load(q_ptr + q_base, mask=mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptr + q_base + HALF, mask=mask, other=0.0).to(tl.float32)
    cos_lo = tl.load(cos_ptr + coeff_base, mask=mask, other=0.0).to(tl.float32)
    cos_hi = tl.load(cos_ptr + coeff_base + HALF, mask=mask, other=0.0).to(tl.float32)
    sin_lo = tl.load(sin_ptr + coeff_base, mask=mask, other=0.0).to(tl.float32)
    sin_hi = tl.load(sin_ptr + coeff_base + HALF, mask=mask, other=0.0).to(tl.float32)

    q_mul_lo = _round_to_bf16_f32(q_lo * cos_lo)
    q_rot_lo = _round_to_bf16_f32((-q_hi) * sin_lo)
    q_mul_hi = _round_to_bf16_f32(q_hi * cos_hi)
    q_rot_hi = _round_to_bf16_f32(q_lo * sin_hi)
    out_q_lo = q_mul_lo + q_rot_lo
    out_q_hi = q_mul_hi + q_rot_hi

    tl.store(
        out_q_ptr + q_base,
        out_q_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )
    tl.store(
        out_q_ptr + q_base + HALF,
        out_q_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )

    kv_mask = mask & (q_head < KVH)
    k_base = ((batch * S + seq) * KVH + q_head) * D + half_d
    k_lo = tl.load(k_ptr + k_base, mask=kv_mask, other=0.0).to(tl.float32)
    k_hi = tl.load(k_ptr + k_base + HALF, mask=kv_mask, other=0.0).to(tl.float32)

    k_mul_lo = _round_to_bf16_f32(k_lo * cos_lo)
    k_rot_lo = _round_to_bf16_f32((-k_hi) * sin_lo)
    k_mul_hi = _round_to_bf16_f32(k_hi * cos_hi)
    k_rot_hi = _round_to_bf16_f32(k_lo * sin_hi)
    out_k_lo = k_mul_lo + k_rot_lo
    out_k_hi = k_mul_hi + k_rot_hi

    tl.store(
        out_k_ptr + k_base,
        out_k_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=kv_mask,
    )
    tl.store(
        out_k_ptr + k_base + HALF,
        out_k_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=kv_mask,
    )

    for group in tl.static_range(0, GROUPS):
        out_head = q_head * GROUPS + group
        repeat_base = ((batch * QH + out_head) * S + seq) * D + half_d
        tl.store(
            out_repeat_ptr + repeat_base,
            out_k_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=kv_mask,
        )
        tl.store(
            out_repeat_ptr + repeat_base + HALF,
            out_k_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=kv_mask,
        )


# 8d7cf075: (T([1000,4096], bf16), T([1,1000,128], bf16), T([1,1000,128], bf16), T([1000,1024], bf16), ...)
@oracle_impl(hardware="B200", point="8d7cf075", BLOCK_PAIRS=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_PAIRS, num_warps):
    q_mm, cos, sin, k_mm = inputs[:4]

    seq = int(cos.shape[1])
    head_dim = int(cos.shape[2])
    half = head_dim // 2
    batch = int(q_mm.shape[0]) // seq
    q_heads = int(q_mm.shape[1]) // head_dim
    kv_heads = int(k_mm.shape[1]) // head_dim
    groups = q_heads // kv_heads

    out_q = torch.empty_strided(
        (batch, q_heads, seq, head_dim),
        (seq * q_heads * head_dim, head_dim, q_heads * head_dim, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )
    out_k = torch.empty_strided(
        (batch, kv_heads, seq, head_dim),
        (seq * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )
    out_repeat = torch.empty_strided(
        (batch, q_heads, seq, head_dim),
        (q_heads * seq * head_dim, seq * head_dim, head_dim, 1),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )

    n_pairs = batch * seq * q_heads * half
    grid = (triton.cdiv(n_pairs, BLOCK_PAIRS),)
    _rotary_q_k_repeat_kernel[grid](
        q_mm,
        cos,
        sin,
        k_mm,
        out_q,
        out_k,
        out_repeat,
        N_PAIRS=n_pairs,
        S=seq,
        QH=q_heads,
        KVH=kv_heads,
        D=head_dim,
        HALF=half,
        GROUPS=groups,
        BLOCK_PAIRS=BLOCK_PAIRS,
        num_warps=num_warps,
    )
    return out_q, out_k, out_repeat
