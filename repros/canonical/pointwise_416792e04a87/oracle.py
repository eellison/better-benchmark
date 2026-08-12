"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-OSS bf16 rotary pointwise scope, including the f32 inverse-frequency/iota phase table, natural cos/sin with the 1.3465735902799727 scale and bf16 cast into the returned strided `[1,1000,32]` tables, the query/key head views, split-half rotary multiply/sub/add sequence with bf16 product rounding, returned contiguous rotated-Q `[64,1000,64]`, returned expanded/clone/permute repeated-K `[64,64,1000]`, and returned last-127-token rotated-K slice view `[1,8,127,64]`; Inductor currently lowers the shared trigonometric table, split/cat rotary producers, grouped-key clone materialization, and sliced side output as separate generic pointwise/layout regions; Inductor cannot do this today because its scheduler cannot keep one shape-derived sin/cos table resident across multiple rotary consumers while also assigning one key producer tile to repeated and sliced returned layouts with alias metadata; the fix is SCHEDULER_FUSION: teach pointwise/layout scheduling to fuse rotary-style trig-table producers through split-half consumers and grouped-key expand/clone users, emitting multi-output tiles with per-output store maps and view-return metadata."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEQ = 1000
Q_HEADS = 64
KV_HEADS = 8
HEAD_DIM = 64
HALF = 32
GROUPS = Q_HEADS // KV_HEADS
TAIL = 127
TAIL_START = SEQ - TAIL
TRIG_NUMEL = SEQ * HALF


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
def _rope_trig_table_kernel(
    inv_freq_ptr,
    out_cos_ptr,
    out_sin_ptr,
    N: tl.constexpr,
    S: tl.constexpr,
    HALF_D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    seq = offsets % S
    half_d = offsets // S
    inv_freq = tl.load(inv_freq_ptr + half_d, mask=mask, other=0.0).to(tl.float32)
    phase = _f32_mul(inv_freq, seq.to(tl.float32))
    cos_value = _f32_mul(libdevice.cos(phase), SCALE).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    sin_value = _f32_mul(libdevice.sin(phase), SCALE).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(out_cos_ptr + offsets, cos_value, mask=mask)
    tl.store(out_sin_ptr + offsets, sin_value, mask=mask)


@triton.jit
def _gptoss_rotary_repeat_tail_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_base_ptr,
    out_repeat_ptr,
    N_PAIRS: tl.constexpr,
    S: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    GROUPS_C: tl.constexpr,
    TAIL_START_C: tl.constexpr,
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
    tail_mask = kv_mask & (seq >= TAIL_START_C)
    tl.store(
        out_k_base_ptr + k_out_base,
        k_out_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=tail_mask,
    )
    tl.store(
        out_k_base_ptr + k_out_base + HALF_D,
        k_out_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=tail_mask,
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


@oracle_impl(
    hardware="B200",
    point="deb9da58",
    TRIG_BLOCK=256,
    BLOCK_PAIRS=1024,
    num_warps=4,
)
def oracle_forward(inputs, *, TRIG_BLOCK: int, BLOCK_PAIRS: int, num_warps: int):
    q_mm, k_mm, inv_freq, *_shape_params = inputs
    out_cos = torch.empty_strided(
        (1, SEQ, HALF),
        (SEQ * HALF, 1, SEQ),
        device=inv_freq.device,
        dtype=torch.bfloat16,
    )
    out_sin = torch.empty_strided(
        (1, SEQ, HALF),
        (SEQ * HALF, 1, SEQ),
        device=inv_freq.device,
        dtype=torch.bfloat16,
    )
    out_q = torch.empty_strided(
        (Q_HEADS, SEQ, HEAD_DIM),
        (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )
    out_k_base = torch.empty_strided(
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

    _rope_trig_table_kernel[(triton.cdiv(TRIG_NUMEL, TRIG_BLOCK),)](
        inv_freq,
        out_cos,
        out_sin,
        N=TRIG_NUMEL,
        S=SEQ,
        HALF_D=HALF,
        SCALE=1.3465735902799727,
        BLOCK=TRIG_BLOCK,
        num_warps=1,
        num_stages=4,
    )

    n_pairs = SEQ * Q_HEADS * HALF
    _gptoss_rotary_repeat_tail_kernel[(triton.cdiv(n_pairs, BLOCK_PAIRS),)](
        q_mm,
        k_mm,
        out_cos,
        out_sin,
        out_q,
        out_k_base,
        out_repeat,
        N_PAIRS=n_pairs,
        S=SEQ,
        QH=Q_HEADS,
        KVH=KV_HEADS,
        D=HEAD_DIM,
        HALF_D=HALF,
        GROUPS_C=GROUPS,
        TAIL_START_C=TAIL_START,
        BLOCK_PAIRS=BLOCK_PAIRS,
        num_warps=num_warps,
        num_stages=3,
    )
    out_k_tail = out_k_base[:, :, TAIL_START:, :]
    return out_cos, out_sin, out_q, out_repeat, out_k_tail
