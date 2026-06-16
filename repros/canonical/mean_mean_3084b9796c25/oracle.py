"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Gemma bf16 dynamic-RoPE query/key RMSNorm scope returned by Repro.forward, including f32 inverse-frequency phase products from the bf16 frequency vector and provided i64 position tensor, natural cos/sin with the captured multiply-by-1 and bf16 table outputs, both fp32 hidden-size-256 RMS reductions with eps=1e-6 and `(weight.float() + 1.0)` affine scale, explicit bf16 rounding before rotary multiplies, the non-contiguous query and key rotary outputs, and the grouped-KV expand/clone/view materialization directly into the contiguous repeated-key output, whereas Inductor lowers the generated RoPE table, two RMS reductions, rotary pointwise branches, and grouped repeat clone as separate generic regions; Inductor cannot do this today because its scheduler cannot keep the generated trigonometric table shared across sibling RMSNorm/RoPE consumers while also assigning one key producer tile to both the repeated-head clone layout and the returned key layout with exact dtype boundaries; the fix is SCHEDULER_FUSION: teach normalization and pointwise/layout scheduling to fuse dynamic RoPE table generation through split-half rotary epilogues and grouped-query repeat materialization with per-output store maps."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 1
SEQ_LEN = 1000
Q_HEADS = 8
KV_HEADS = 4
HEAD_DIM = 256
HALF_DIM = 128
EPS = 1.0e-6
Q_ROWS = BATCH * SEQ_LEN * Q_HEADS
KV_ROWS = BATCH * SEQ_LEN * KV_HEADS
REPEAT = Q_HEADS // KV_HEADS
TRIG_NUMEL = BATCH * SEQ_LEN * HEAD_DIM


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
def _round_bf16_f32(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _positions_trig_kernel(
    inv_freq_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    N_ELEMENTS: tl.constexpr,
    SEQ: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N_ELEMENTS

    dim = offsets % D
    seq = (offsets // D) % SEQ
    freq_idx = dim % HALF
    inv_freq = tl.load(inv_freq_ptr + freq_idx, mask=mask, other=0.0).to(tl.float32)
    pos = tl.load(positions_ptr + seq, mask=mask, other=0).to(tl.float32)
    phase = _f32_mul(inv_freq, pos)
    cos_value = _f32_mul(libdevice.cos(phase), 1.0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    sin_value = _f32_mul(libdevice.sin(phase), 1.0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(cos_ptr + offsets, cos_value, mask=mask)
    tl.store(sin_ptr + offsets, sin_value, mask=mask)


@triton.jit
def _rmsnorm_rope_block(
    mm_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    rows,
    cols,
    row_mask,
    NUM_HEADS: tl.constexpr,
    SEQ: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    EPS_VALUE: tl.constexpr,
):
    mask = row_mask[:, None]
    offsets = rows[:, None] * D + cols[None, :]
    x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(tl.where(mask, _f32_mul(x, x), 0.0), axis=1)
    mean_sq = _f32_mul(sum_sq, 1.0 / D)
    inv_rms = tl.rsqrt(_f32_add(mean_sq, EPS_VALUE))

    rot_cols = tl.where(cols < HALF, cols + HALF, cols - HALF)
    x_rot = tl.load(
        mm_ptr + rows[:, None] * D + rot_cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = _f32_add(tl.load(weight_ptr + cols).to(tl.float32), 1.0)
    rot_weight = _f32_add(tl.load(weight_ptr + rot_cols).to(tl.float32), 1.0)
    value = _round_bf16_f32(_f32_mul(_f32_mul(x, inv_rms[:, None]), weight[None, :]))
    rot_value = _round_bf16_f32(
        _f32_mul(_f32_mul(x_rot, inv_rms[:, None]), rot_weight[None, :])
    )
    rot_value = _f32_mul(
        rot_value,
        tl.where(cols < HALF, -1.0, 1.0)[None, :],
    )

    pos = (rows // NUM_HEADS) - ((rows // (SEQ * NUM_HEADS)) * SEQ)
    rope_offsets = pos[:, None] * D + cols[None, :]
    cos_value = tl.load(cos_ptr + rope_offsets, mask=mask, other=0.0).to(tl.float32)
    sin_value = tl.load(sin_ptr + rope_offsets, mask=mask, other=0.0).to(tl.float32)
    direct = _round_bf16_f32(_f32_mul(value, cos_value))
    rotated = _round_bf16_f32(_f32_mul(rot_value, sin_value))
    out = _f32_add(direct, rotated)
    return out, pos


@triton.jit
def _gemma_dynamic_rmsnorm_rope_kernel(
    q_ptr,
    q_weight_ptr,
    cos_ptr,
    sin_ptr,
    k_ptr,
    k_weight_ptr,
    q_out_ptr,
    k_out_ptr,
    repeat_out_ptr,
    Q_N_ROWS: tl.constexpr,
    K_N_ROWS: tl.constexpr,
    Q_NUM_HEADS: tl.constexpr,
    K_NUM_HEADS: tl.constexpr,
    SEQ: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    REPEAT_COUNT: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    base_rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, D)

    q_rows_0 = base_rows
    q_mask_0 = q_rows_0 < K_N_ROWS
    q_out_0, _ = _rmsnorm_rope_block(
        q_ptr,
        q_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_rows_0,
        cols,
        q_mask_0,
        Q_NUM_HEADS,
        SEQ,
        D,
        HALF,
        EPS_VALUE,
    )
    tl.store(
        q_out_ptr + q_rows_0[:, None] * D + cols[None, :],
        q_out_0.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=q_mask_0[:, None],
    )

    q_rows_1 = base_rows + K_N_ROWS
    q_mask_1 = q_rows_1 < Q_N_ROWS
    q_out_1, _ = _rmsnorm_rope_block(
        q_ptr,
        q_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_rows_1,
        cols,
        q_mask_1,
        Q_NUM_HEADS,
        SEQ,
        D,
        HALF,
        EPS_VALUE,
    )
    tl.store(
        q_out_ptr + q_rows_1[:, None] * D + cols[None, :],
        q_out_1.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=q_mask_1[:, None],
    )

    k_rows = base_rows
    k_mask = k_rows < K_N_ROWS
    k_out, pos = _rmsnorm_rope_block(
        k_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        k_rows,
        cols,
        k_mask,
        K_NUM_HEADS,
        SEQ,
        D,
        HALF,
        EPS_VALUE,
    )
    k_out_bf16 = k_out.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(
        k_out_ptr + k_rows[:, None] * D + cols[None, :],
        k_out_bf16,
        mask=k_mask[:, None],
    )

    kv_head = k_rows - (k_rows // K_NUM_HEADS) * K_NUM_HEADS
    batch = k_rows // (SEQ * K_NUM_HEADS)
    for repeat_idx in tl.static_range(0, REPEAT_COUNT):
        out_head = kv_head * REPEAT_COUNT + repeat_idx
        repeat_offsets = (
            batch[:, None] * (Q_NUM_HEADS * SEQ * D)
            + out_head[:, None] * (SEQ * D)
            + pos[:, None] * D
            + cols[None, :]
        )
        tl.store(repeat_out_ptr + repeat_offsets, k_out_bf16, mask=k_mask[:, None])


@oracle_impl(
    hardware="B200",
    point="5942af01",
    TRIG_BLOCK=256,
    BLOCK_ROWS=2,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    TRIG_BLOCK: int,
    BLOCK_ROWS: int,
    num_warps: int,
):
    q_mm, q_weight, inv_freq, positions, k_mm, k_weight = inputs[:6]

    cos = torch.empty_strided(
        (BATCH, SEQ_LEN, HEAD_DIM),
        (SEQ_LEN * HEAD_DIM, HEAD_DIM, 1),
        device=inv_freq.device,
        dtype=torch.bfloat16,
    )
    sin = torch.empty_strided(
        (BATCH, SEQ_LEN, HEAD_DIM),
        (SEQ_LEN * HEAD_DIM, HEAD_DIM, 1),
        device=inv_freq.device,
        dtype=torch.bfloat16,
    )
    query = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ_LEN, HEAD_DIM),
        (Q_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )
    key = torch.empty_strided(
        (BATCH, KV_HEADS, SEQ_LEN, HEAD_DIM),
        (KV_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, KV_HEADS * HEAD_DIM, 1),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )
    repeated_key = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ_LEN, HEAD_DIM),
        (Q_HEADS * SEQ_LEN * HEAD_DIM, SEQ_LEN * HEAD_DIM, HEAD_DIM, 1),
        device=k_mm.device,
        dtype=torch.bfloat16,
    )

    _positions_trig_kernel[(triton.cdiv(TRIG_NUMEL, TRIG_BLOCK),)](
        inv_freq,
        positions,
        cos,
        sin,
        N_ELEMENTS=TRIG_NUMEL,
        SEQ=SEQ_LEN,
        D=HEAD_DIM,
        HALF=HALF_DIM,
        BLOCK=TRIG_BLOCK,
        num_warps=1,
        num_stages=4,
    )
    _gemma_dynamic_rmsnorm_rope_kernel[(triton.cdiv(KV_ROWS, BLOCK_ROWS),)](
        q_mm,
        q_weight,
        cos,
        sin,
        k_mm,
        k_weight,
        query,
        key,
        repeated_key,
        Q_N_ROWS=Q_ROWS,
        K_N_ROWS=KV_ROWS,
        Q_NUM_HEADS=Q_HEADS,
        K_NUM_HEADS=KV_HEADS,
        SEQ=SEQ_LEN,
        D=HEAD_DIM,
        HALF=HALF_DIM,
        REPEAT_COUNT=REPEAT,
        EPS_VALUE=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=num_warps,
        num_stages=3,
    )
    return cos, sin, query, key, repeated_key
