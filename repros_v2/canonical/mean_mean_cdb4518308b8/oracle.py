"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Qwen3 bf16 attention-projection epilogue returned by Repro.forward, including generated RoPE cos/sin tables from inverse frequency and positions, query RMSNorm over `[1,1000,16,128]`, query rotate-half application into the returned permuted layout, key RMSNorm over `[1,1000,8,128]`, key RoPE, and grouped-KV expand/clone/view materialization into the returned contiguous repeated-head output, whereas Inductor lowers the generated trigonometric table, two RMSNorm reductions, rotate-half slice/cat branches, layout transforms, and KV repeat clone as generic scheduled regions; Inductor cannot do this today because its scheduler and pattern library do not recognize this Qwen RMSNorm-plus-generated-RoPE epilogue with shared tables and grouped-query KV expansion as one full-scope semantic lowering; the fix is NEW_PATTERN: add a guarded Qwen attention-projection epilogue template that fuses fixed-hidden RMSNorm, generated rotary embedding, layout-aware Q/K writes, and repeat-kv materialization while preserving bf16 cast boundaries and returned strides."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEQ = 1000
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
HALF = 64
GROUPS = Q_HEADS // KV_HEADS
TABLE_NUMEL = SEQ * HEAD_DIM
Q_ROWS = SEQ * Q_HEADS
K_ROWS = SEQ * KV_HEADS
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
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    dim = offsets % D
    seq = offsets // D
    inv_freq = tl.load(inv_freq_ptr + (dim % HALF_D), mask=mask, other=0.0).to(
        tl.float32
    )
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
def _rms_rope_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, D)
    row_mask = rows < N_ROWS
    mask = row_mask[:, None]

    base = rows[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(tl.where(mask, _f32_mul(x, x), 0.0), axis=1)
    mean_sq = _f32_mul(sum_sq, 1.0 / D)
    inv_rms = tl.rsqrt(_f32_add(mean_sq, EPS_VALUE))

    rot_cols = tl.where(cols < HALF_D, cols + HALF_D, cols - HALF_D)
    x_rot = tl.load(
        x_ptr + rows[:, None] * D + rot_cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    rot_weight = tl.load(weight_ptr + rot_cols).to(tl.float32)
    normalized = _round_to_bf16_f32(_f32_mul(x, inv_rms[:, None]))
    normalized_rot = _round_to_bf16_f32(_f32_mul(x_rot, inv_rms[:, None]))
    value = _round_to_bf16_f32(_f32_mul(normalized, weight[None, :]))
    rot_value = _round_to_bf16_f32(_f32_mul(normalized_rot, rot_weight[None, :]))
    rot_value = _f32_mul(rot_value, tl.where(cols < HALF_D, -1.0, 1.0)[None, :])

    pos = (rows // NUM_HEADS) - ((rows // (SEQ_LEN * NUM_HEADS)) * SEQ_LEN)
    rope_base = pos[:, None] * D + cols[None, :]
    cos_value = tl.load(cos_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)
    sin_value = tl.load(sin_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)

    direct = _round_to_bf16_f32(_f32_mul(value, cos_value))
    rotated = _round_to_bf16_f32(_f32_mul(rot_value, sin_value))
    out = _f32_add(direct, rotated)
    tl.store(out_ptr + base, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@triton.jit
def _k_rms_rope_repeat_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    k_out_ptr,
    repeat_out_ptr,
    N_ROWS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    OUT_HEADS: tl.constexpr,
    REPEAT: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, D)
    row_mask = rows < N_ROWS
    mask = row_mask[:, None]

    base = rows[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(tl.where(mask, _f32_mul(x, x), 0.0), axis=1)
    mean_sq = _f32_mul(sum_sq, 1.0 / D)
    inv_rms = tl.rsqrt(_f32_add(mean_sq, EPS_VALUE))

    rot_cols = tl.where(cols < HALF_D, cols + HALF_D, cols - HALF_D)
    x_rot = tl.load(
        x_ptr + rows[:, None] * D + rot_cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    rot_weight = tl.load(weight_ptr + rot_cols).to(tl.float32)
    normalized = _round_to_bf16_f32(_f32_mul(x, inv_rms[:, None]))
    normalized_rot = _round_to_bf16_f32(_f32_mul(x_rot, inv_rms[:, None]))
    value = _round_to_bf16_f32(_f32_mul(normalized, weight[None, :]))
    rot_value = _round_to_bf16_f32(_f32_mul(normalized_rot, rot_weight[None, :]))
    rot_value = _f32_mul(rot_value, tl.where(cols < HALF_D, -1.0, 1.0)[None, :])

    pos = (rows // NUM_HEADS) - ((rows // (SEQ_LEN * NUM_HEADS)) * SEQ_LEN)
    rope_base = pos[:, None] * D + cols[None, :]
    cos_value = tl.load(cos_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)
    sin_value = tl.load(sin_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)

    direct = _round_to_bf16_f32(_f32_mul(value, cos_value))
    rotated = _round_to_bf16_f32(_f32_mul(rot_value, sin_value))
    out = _f32_add(direct, rotated)
    out_bf16 = out.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(k_out_ptr + base, out_bf16, mask=mask)

    kv_head = rows - (rows // NUM_HEADS) * NUM_HEADS
    batch = rows // (SEQ_LEN * NUM_HEADS)
    for repeat_idx in tl.static_range(0, REPEAT):
        out_head = kv_head * REPEAT + repeat_idx
        repeat_base = (
            batch[:, None] * (OUT_HEADS * SEQ_LEN * D)
            + out_head[:, None] * (SEQ_LEN * D)
            + pos[:, None] * D
            + cols[None, :]
        )
        tl.store(repeat_out_ptr + repeat_base, out_bf16, mask=mask)


@oracle_impl(
    hardware="B200",
    point="b0137972",
    TABLE_BLOCK=256,
    BLOCK_ROWS=4,
    num_warps=4,
)
def oracle_forward(inputs, *, TABLE_BLOCK: int, BLOCK_ROWS: int, num_warps: int):
    q_mm, q_weight, inv_freq, k_mm, k_weight, *_shape_params = inputs

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
        D=HEAD_DIM,
        HALF_D=HALF,
        BLOCK=TABLE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _rms_rope_kernel[(triton.cdiv(Q_ROWS, BLOCK_ROWS),)](
        q_mm,
        q_weight,
        cos_out,
        sin_out,
        q_out,
        N_ROWS=Q_ROWS,
        NUM_HEADS=Q_HEADS,
        SEQ_LEN=SEQ,
        D=HEAD_DIM,
        HALF_D=HALF,
        EPS_VALUE=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=num_warps,
        num_stages=3,
    )
    _k_rms_rope_repeat_kernel[(triton.cdiv(K_ROWS, BLOCK_ROWS),)](
        k_mm,
        k_weight,
        cos_out,
        sin_out,
        k_out,
        repeat_out,
        N_ROWS=K_ROWS,
        NUM_HEADS=KV_HEADS,
        OUT_HEADS=Q_HEADS,
        REPEAT=GROUPS,
        SEQ_LEN=SEQ,
        D=HEAD_DIM,
        HALF_D=HALF,
        EPS_VALUE=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=num_warps,
        num_stages=3,
    )
    return cos_out, sin_out, q_out, k_out, repeat_out
