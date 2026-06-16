"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete GPT-OSS bf16 rotary pointwise scope, including the query/key head views, split-half rotary multiply/sub/add sequence with bf16 product rounding, returned contiguous rotated-Q `[64,1000,64]`, returned contiguous rotated-K `[1,8,1000,64]`, and the expanded/clone/permute repeated-K `[64,64,1000]` layout; Inductor currently lowers the split/cat rotary producers and grouped-key clone materialization as separate generic pointwise and layout regions; Inductor cannot do this today because its scheduler does not assign one rotary producer tile to multiple returned layouts plus an expanded clone side output; the fix is SCHEDULER_FUSION: teach pointwise/layout scheduling to fuse rotary-style split-half producers through grouped-key expand/clone consumers and emit multi-output tiles with per-output store maps."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 1000
Q_HEADS = 64
KV_HEADS = 8
HEAD_DIM = 64
HALF = 32
GROUPS = Q_HEADS // KV_HEADS


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
def _gptoss_rotary_repeat_kernel(
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

    coeff_offsets = half_d * S + seq
    cos = tl.load(cos_ptr + coeff_offsets, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + coeff_offsets, mask=mask, other=0.0).to(tl.float32)

    q_in_base = seq * QH * D + q_head * D + half_d
    q_lo = tl.load(q_ptr + q_in_base, mask=mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptr + q_in_base + HALF_D, mask=mask, other=0.0).to(tl.float32)

    q_mul_lo = _round_to_bf16_f32(q_lo * cos)
    q_rot_lo = _round_to_bf16_f32(q_hi * sin)
    q_mul_hi = _round_to_bf16_f32(q_hi * cos)
    q_rot_hi = _round_to_bf16_f32(q_lo * sin)
    q_out_lo = q_mul_lo - q_rot_lo
    q_out_hi = q_mul_hi + q_rot_hi

    q_out_base = q_head * S * D + seq * D + half_d
    tl.store(
        out_q_ptr + q_out_base,
        q_out_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )
    tl.store(
        out_q_ptr + q_out_base + HALF_D,
        q_out_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )

    kv_mask = mask & (q_head < KVH)
    k_in_base = seq * KVH * D + q_head * D + half_d
    k_lo = tl.load(k_ptr + k_in_base, mask=kv_mask, other=0.0).to(tl.float32)
    k_hi = tl.load(k_ptr + k_in_base + HALF_D, mask=kv_mask, other=0.0).to(tl.float32)

    k_mul_lo = _round_to_bf16_f32(k_lo * cos)
    k_rot_lo = _round_to_bf16_f32(k_hi * sin)
    k_mul_hi = _round_to_bf16_f32(k_hi * cos)
    k_rot_hi = _round_to_bf16_f32(k_lo * sin)
    k_out_lo = k_mul_lo - k_rot_lo
    k_out_hi = k_mul_hi + k_rot_hi

    k_out_base = q_head * S * D + seq * D + half_d
    tl.store(
        out_k_ptr + k_out_base,
        k_out_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=kv_mask,
    )
    tl.store(
        out_k_ptr + k_out_base + HALF_D,
        k_out_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=kv_mask,
    )

    for group in tl.static_range(0, GROUPS_C):
        repeat_head = q_head * GROUPS_C + group
        repeat_base = repeat_head * S * D + half_d + seq * D
        tl.store(
            out_repeat_ptr + repeat_base,
            k_out_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=kv_mask,
        )
        tl.store(
            out_repeat_ptr + repeat_base + HALF_D,
            k_out_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=kv_mask,
        )


@oracle_impl(hardware="B200", point="fa3cc74f", BLOCK_PAIRS=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_PAIRS: int, num_warps: int):
    q_mm, k_mm, cos, sin, *_shape_params = inputs
    out_q = torch.empty_strided(
        (Q_HEADS, SEQ, HEAD_DIM),
        (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )
    out_k = torch.empty_strided(
        (1, KV_HEADS, SEQ, HEAD_DIM),
        (KV_HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )
    out_repeat = torch.empty_strided(
        (Q_HEADS, HEAD_DIM, SEQ),
        (SEQ * HEAD_DIM, 1, HEAD_DIM),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )

    n_pairs = SEQ * Q_HEADS * HALF
    _gptoss_rotary_repeat_kernel[(triton.cdiv(n_pairs, BLOCK_PAIRS),)](
        q_mm,
        k_mm,
        cos,
        sin,
        out_q,
        out_k,
        out_repeat,
        N_PAIRS=n_pairs,
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
    return out_q, out_k, out_repeat
