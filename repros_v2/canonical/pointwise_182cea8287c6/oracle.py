"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Gemma dynamic RoPE pointwise scope with one trigonometric table kernel and one layout-aware rotary/repeat kernel, including the f32 iota phase product from the bf16 inverse-frequency vector, natural cos/sin, bf16 table outputs, bf16-rounded query/key rotary multiplies and adds, grouped-key expand/clone/view materialization, and the full key slice return, whereas Inductor lowers the generated cos/sin table, rotate-half slice/cat arithmetic, non-contiguous query/key stores, repeated-key clone, and slice side output as separate generic pointwise/layout regions; Inductor cannot do this today because its scheduler cannot keep a generated RoPE table producer live across sibling rotary consumers while also assigning one key producer tile to both repeated and sliced returned layouts with exact dtype boundaries; the fix is SCHEDULER_FUSION: teach pointwise/layout scheduling to fuse generated RoPE tables through split-half rotary consumers and grouped-key repeat materialization with per-output store maps and view-return metadata."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 1
SEQ = 1000
Q_HEADS = 8
K_HEADS = 4
HEAD_DIM = 256
HALF = 128
GROUPS = Q_HEADS // K_HEADS
TRIG_NUMEL = BATCH * SEQ * HEAD_DIM


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
def _rope_trig_kernel(
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    n_elements: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    dim = offsets % head_dim
    seq = (offsets // head_dim) % seq_len
    freq = dim % half_dim
    inv_freq = tl.load(inv_freq_ptr + freq, mask=mask, other=0.0).to(tl.float32)
    phase = _f32_mul(inv_freq, seq.to(tl.float32))
    cos_value = libdevice.cos(phase).to(tl.bfloat16, fp_downcast_rounding="rtne")
    sin_value = libdevice.sin(phase).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(cos_ptr + offsets, cos_value, mask=mask)
    tl.store(sin_ptr + offsets, sin_value, mask=mask)


@triton.jit
def _rotary_q_k_repeat_kernel(
    q_ptr,
    cos_ptr,
    sin_ptr,
    k_ptr,
    out_q_ptr,
    out_k_ptr,
    out_repeat_ptr,
    n_pairs: tl.constexpr,
    seq_len: tl.constexpr,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    pair_offsets = tl.program_id(0) * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    mask = pair_offsets < n_pairs

    half_d = pair_offsets % half_dim
    q_head = (pair_offsets // half_dim) % q_heads
    seq = (pair_offsets // (half_dim * q_heads)) % seq_len
    batch = pair_offsets // (half_dim * q_heads * seq_len)

    q_base = ((batch * seq_len + seq) * q_heads + q_head) * head_dim + half_d
    trig_base = (batch * seq_len + seq) * head_dim + half_d

    q_lo = tl.load(q_ptr + q_base, mask=mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptr + q_base + half_dim, mask=mask, other=0.0).to(tl.float32)
    cos_lo = tl.load(cos_ptr + trig_base, mask=mask, other=0.0).to(tl.float32)
    cos_hi = tl.load(cos_ptr + trig_base + half_dim, mask=mask, other=0.0).to(
        tl.float32
    )
    sin_lo = tl.load(sin_ptr + trig_base, mask=mask, other=0.0).to(tl.float32)
    sin_hi = tl.load(sin_ptr + trig_base + half_dim, mask=mask, other=0.0).to(
        tl.float32
    )

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
        out_q_ptr + q_base + half_dim,
        out_q_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )

    k_mask = mask & (q_head < k_heads)
    k_base = ((batch * seq_len + seq) * k_heads + q_head) * head_dim + half_d
    k_lo = tl.load(k_ptr + k_base, mask=k_mask, other=0.0).to(tl.float32)
    k_hi = tl.load(k_ptr + k_base + half_dim, mask=k_mask, other=0.0).to(tl.float32)

    k_mul_lo = _round_to_bf16_f32(k_lo * cos_lo)
    k_rot_lo = _round_to_bf16_f32((-k_hi) * sin_lo)
    k_mul_hi = _round_to_bf16_f32(k_hi * cos_hi)
    k_rot_hi = _round_to_bf16_f32(k_lo * sin_hi)
    out_k_lo = k_mul_lo + k_rot_lo
    out_k_hi = k_mul_hi + k_rot_hi

    tl.store(
        out_k_ptr + k_base,
        out_k_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=k_mask,
    )
    tl.store(
        out_k_ptr + k_base + half_dim,
        out_k_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=k_mask,
    )

    for group in tl.static_range(0, groups):
        out_head = q_head * groups + group
        repeat_base = ((batch * q_heads + out_head) * seq_len + seq) * head_dim + half_d
        tl.store(
            out_repeat_ptr + repeat_base,
            out_k_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=k_mask,
        )
        tl.store(
            out_repeat_ptr + repeat_base + half_dim,
            out_k_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=k_mask,
        )


# 5da3d9cf: Gemma-2 dynamic cos/sin RoPE, q bf16[1000,2048], k bf16[1000,1024]
@oracle_impl(
    hardware="B200",
    point="5da3d9cf",
    TRIG_BLOCK=256,
    BLOCK_PAIRS=256,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    TRIG_BLOCK: int,
    BLOCK_PAIRS: int,
    num_warps: int,
):
    q_mm, inv_freq, k_mm, *_shape_params = inputs
    cos = torch.empty_strided(
        (BATCH, SEQ, HEAD_DIM),
        (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=inv_freq.device,
        dtype=torch.bfloat16,
    )
    sin = torch.empty_strided(
        (BATCH, SEQ, HEAD_DIM),
        (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=inv_freq.device,
        dtype=torch.bfloat16,
    )
    out_q = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ, HEAD_DIM),
        (Q_HEADS * SEQ * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )
    out_repeat = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ, HEAD_DIM),
        (Q_HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )
    out_k = torch.empty_strided(
        (BATCH, K_HEADS, SEQ, HEAD_DIM),
        (K_HEADS * SEQ * HEAD_DIM, HEAD_DIM, K_HEADS * HEAD_DIM, 1),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )

    _rope_trig_kernel[(triton.cdiv(TRIG_NUMEL, TRIG_BLOCK),)](
        inv_freq,
        cos,
        sin,
        n_elements=TRIG_NUMEL,
        seq_len=SEQ,
        head_dim=HEAD_DIM,
        half_dim=HALF,
        BLOCK=TRIG_BLOCK,
        num_warps=1,
        num_stages=4,
    )

    n_pairs = BATCH * SEQ * Q_HEADS * HALF
    _rotary_q_k_repeat_kernel[(triton.cdiv(n_pairs, BLOCK_PAIRS),)](
        q_mm,
        cos,
        sin,
        k_mm,
        out_q,
        out_k,
        out_repeat,
        n_pairs=n_pairs,
        seq_len=SEQ,
        q_heads=Q_HEADS,
        k_heads=K_HEADS,
        head_dim=HEAD_DIM,
        half_dim=HALF,
        groups=GROUPS,
        BLOCK_PAIRS=BLOCK_PAIRS,
        num_warps=num_warps,
        num_stages=3,
    )
    return cos, sin, out_q, out_repeat, out_k
