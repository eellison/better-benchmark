"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Gemma query/key RMSNorm plus RoPE scope, including the bf16 rounded normalization, query non-contiguous output, grouped-KV expand/clone/view materialization, and returned key slice view, whereas Inductor schedules the two reductions and grouped-key materialization as generic separate kernels; Inductor cannot do this today because its reduction scheduler does not sink fixed-hidden RoPE epilogues and repeated grouped-KV stores into the normalization producer while preserving the visible layouts; the fix is SCHEDULER_FUSION: add a guarded RMSNorm-RoPE grouped-KV lowering that writes the query, repeated key, and key-slice layouts directly from the row-normalization kernels."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


HEAD_DIM = 256
HALF_DIM = 128
SEQ_LEN = 1000
Q_HEADS = 8
KV_HEADS = 4
REPEAT = 2
EPS = 1.0e-6


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
def _round_bf16(x):
    return tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;",
        constraints="=h,f",
        args=[x],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _query_rmsnorm_rope_kernel(
    mm_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_D)
    pair_cols = tl.arange(0, BLOCK_HALF)
    mask = rows[:, None] < (SEQ * HEADS)
    base = rows[:, None] * BLOCK_D + cols[None, :]

    x = tl.load(mm_ptr + base, mask=mask, other=0.0).to(tl.float32)
    even_x = tl.load(mm_ptr + rows[:, None] * BLOCK_D + (pair_cols[None, :] * 2), mask=rows[:, None] < (SEQ * HEADS), other=0.0).to(tl.float32)
    odd_x = tl.load(mm_ptr + rows[:, None] * BLOCK_D + (pair_cols[None, :] * 2 + 1), mask=rows[:, None] < (SEQ * HEADS), other=0.0).to(tl.float32)
    pair_squares = _f32_add(_f32_mul(even_x, even_x), _f32_mul(odd_x, odd_x))
    sum_sq = tl.sum(tl.where(rows[:, None] < (SEQ * HEADS), pair_squares, 0.0), axis=1)
    inv_rms = libdevice.rsqrt(_f32_add(_f32_mul(sum_sq, 1.0 / BLOCK_D), EPS_VALUE))

    rot_cols = tl.where(cols < BLOCK_HALF, cols + BLOCK_HALF, cols - BLOCK_HALF)
    x_rot = tl.load(
        mm_ptr + rows[:, None] * BLOCK_D + rot_cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = _f32_add(tl.load(weight_ptr + cols).to(tl.float32), 1.0)
    rot_weight = _f32_add(tl.load(weight_ptr + rot_cols).to(tl.float32), 1.0)
    q = _round_bf16(_f32_mul(_f32_mul(x, inv_rms[:, None]), weight[None, :]))
    q_rot = _round_bf16(_f32_mul(_f32_mul(x_rot, inv_rms[:, None]), rot_weight[None, :]))
    q_rot = tl.where(cols < BLOCK_HALF, -q_rot, q_rot).to(tl.bfloat16)

    seq = rows // HEADS
    rope_offsets = seq[:, None] * BLOCK_D + cols[None, :]
    cos = tl.load(cos_ptr + rope_offsets, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + rope_offsets, mask=mask, other=0.0)

    prod = _round_bf16(_f32_mul(q.to(tl.float32), cos.to(tl.float32)))
    rot_prod = _round_bf16(_f32_mul(q_rot.to(tl.float32), sin.to(tl.float32)))
    out = _round_bf16(_f32_add(prod.to(tl.float32), rot_prod.to(tl.float32)))
    tl.store(out_ptr + base, out, mask=mask)


@triton.jit
def _key_rmsnorm_rope_repeat_kernel(
    mm_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    repeat_out_ptr,
    slice_out_ptr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    REPEAT_VALUE: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_D)
    pair_cols = tl.arange(0, BLOCK_HALF)
    mask = rows[:, None] < (SEQ * HEADS)
    base = rows[:, None] * BLOCK_D + cols[None, :]

    x = tl.load(mm_ptr + base, mask=mask, other=0.0).to(tl.float32)
    even_x = tl.load(mm_ptr + rows[:, None] * BLOCK_D + (pair_cols[None, :] * 2), mask=rows[:, None] < (SEQ * HEADS), other=0.0).to(tl.float32)
    odd_x = tl.load(mm_ptr + rows[:, None] * BLOCK_D + (pair_cols[None, :] * 2 + 1), mask=rows[:, None] < (SEQ * HEADS), other=0.0).to(tl.float32)
    pair_squares = _f32_add(_f32_mul(even_x, even_x), _f32_mul(odd_x, odd_x))
    sum_sq = tl.sum(tl.where(rows[:, None] < (SEQ * HEADS), pair_squares, 0.0), axis=1)
    inv_rms = libdevice.rsqrt(_f32_add(_f32_mul(sum_sq, 1.0 / BLOCK_D), EPS_VALUE))

    rot_cols = tl.where(cols < BLOCK_HALF, cols + BLOCK_HALF, cols - BLOCK_HALF)
    x_rot = tl.load(
        mm_ptr + rows[:, None] * BLOCK_D + rot_cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = _f32_add(tl.load(weight_ptr + cols).to(tl.float32), 1.0)
    rot_weight = _f32_add(tl.load(weight_ptr + rot_cols).to(tl.float32), 1.0)
    k = _round_bf16(_f32_mul(_f32_mul(x, inv_rms[:, None]), weight[None, :]))
    k_rot = _round_bf16(_f32_mul(_f32_mul(x_rot, inv_rms[:, None]), rot_weight[None, :]))
    k_rot = tl.where(cols < BLOCK_HALF, -k_rot, k_rot).to(tl.bfloat16)

    seq = rows // HEADS
    kv_head = rows - seq * HEADS
    rope_offsets = seq[:, None] * BLOCK_D + cols[None, :]
    cos = tl.load(cos_ptr + rope_offsets, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + rope_offsets, mask=mask, other=0.0)

    prod = _round_bf16(_f32_mul(k.to(tl.float32), cos.to(tl.float32)))
    rot_prod = _round_bf16(_f32_mul(k_rot.to(tl.float32), sin.to(tl.float32)))
    out = _round_bf16(_f32_add(prod.to(tl.float32), rot_prod.to(tl.float32)))
    tl.store(slice_out_ptr + base, out, mask=mask)

    for repeat_idx in tl.static_range(0, REPEAT_VALUE):
        out_head = kv_head * REPEAT_VALUE + repeat_idx
        repeat_offsets = out_head[:, None] * (SEQ * BLOCK_D) + seq[:, None] * BLOCK_D + cols[None, :]
        tl.store(repeat_out_ptr + repeat_offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="2c02d9cc", BLOCK_ROWS=4, BLOCK_D=HEAD_DIM, BLOCK_HALF=HALF_DIM)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_D, BLOCK_HALF):
    q_proj, q_weight, cos, sin, k_proj, k_weight = inputs[:6]

    q_out = torch.empty_strided(
        (1, Q_HEADS, SEQ_LEN, HEAD_DIM),
        (Q_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1),
        device=q_proj.device,
        dtype=torch.bfloat16,
    )
    k_repeat = torch.empty(
        (1, Q_HEADS, SEQ_LEN, HEAD_DIM),
        device=q_proj.device,
        dtype=torch.bfloat16,
    )
    k_slice = torch.empty_strided(
        (1, KV_HEADS, SEQ_LEN, HEAD_DIM),
        (KV_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, KV_HEADS * HEAD_DIM, 1),
        device=q_proj.device,
        dtype=torch.bfloat16,
    )

    _query_rmsnorm_rope_kernel[(triton.cdiv(SEQ_LEN * Q_HEADS, BLOCK_ROWS),)](
        q_proj,
        q_weight,
        cos,
        sin,
        q_out,
        SEQ=SEQ_LEN,
        HEADS=Q_HEADS,
        EPS_VALUE=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_D=BLOCK_D,
        BLOCK_HALF=BLOCK_HALF,
        num_warps=8,
        num_stages=3,
    )
    _key_rmsnorm_rope_repeat_kernel[(triton.cdiv(SEQ_LEN * KV_HEADS, BLOCK_ROWS),)](
        k_proj,
        k_weight,
        cos,
        sin,
        k_repeat,
        k_slice,
        SEQ=SEQ_LEN,
        HEADS=KV_HEADS,
        REPEAT_VALUE=REPEAT,
        EPS_VALUE=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_D=BLOCK_D,
        BLOCK_HALF=BLOCK_HALF,
        num_warps=8,
        num_stages=3,
    )
    return q_out, k_repeat, k_slice
