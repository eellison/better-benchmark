"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Gemma bf16 query/key RMSNorm plus precomputed-RoPE scope returned by Repro.forward, including the fp32 RMS reductions over hidden size 256, the captured `(weight + 1.0)` affine scale, bf16 rounding before RoPE multiplies, non-contiguous query and key RoPE outputs, and the grouped-KV expand/clone/view materialization directly into the contiguous repeated-head output, whereas Inductor lowers the same work as generic reduction kernels followed by separate RoPE pointwise/layout and expand-clone materialization kernels; Inductor cannot do this today because the fixed-hidden normalization scheduler does not sink the rotate-half RoPE epilogue and grouped-KV repeat side stores into one multi-output row plan while preserving each returned output stride; the fix is SCHEDULER_FUSION: teach normalization scheduling to fuse static RoPE consumers and grouped-query repeat materialization into the producer epilogue with per-output layout stores."""

import torch
import triton
import triton.language as tl

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
    return x.to(tl.bfloat16).to(tl.float32)


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
    sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=1)
    inv_rms = tl.rsqrt(sum_sq * (1.0 / D) + EPS_VALUE)

    rot_cols = tl.where(cols < HALF, cols + HALF, cols - HALF)
    x_rot = tl.load(
        mm_ptr + rows[:, None] * D + rot_cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(weight_ptr + cols).to(tl.float32) + 1.0
    rot_weight = tl.load(weight_ptr + rot_cols).to(tl.float32) + 1.0
    value = _round_bf16(_f32_mul(_f32_mul(x, inv_rms[:, None]), weight[None, :]))
    rot_value = _round_bf16(
        _f32_mul(_f32_mul(x_rot, inv_rms[:, None]), rot_weight[None, :])
    )
    rot_value = rot_value * tl.where(cols < HALF, -1.0, 1.0)[None, :]

    pos = (rows // NUM_HEADS) - ((rows // (SEQ * NUM_HEADS)) * SEQ)
    rope_offsets = pos[:, None] * D + cols[None, :]
    cos_value = tl.load(cos_ptr + rope_offsets, mask=mask, other=0.0).to(tl.float32)
    sin_value = tl.load(sin_ptr + rope_offsets, mask=mask, other=0.0).to(tl.float32)
    out = _round_bf16(_f32_mul(value, cos_value)) + _round_bf16(
        _f32_mul(rot_value, sin_value)
    )
    return out, pos


@triton.jit
def _gemma_rmsnorm_rope_fused_kernel(
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
    tl.store(q_out_ptr + q_rows_0[:, None] * D + cols[None, :], q_out_0, mask=q_mask_0[:, None])

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
    tl.store(q_out_ptr + q_rows_1[:, None] * D + cols[None, :], q_out_1, mask=q_mask_1[:, None])

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
    tl.store(k_out_ptr + k_rows[:, None] * D + cols[None, :], k_out, mask=k_mask[:, None])

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
        tl.store(repeat_out_ptr + repeat_offsets, k_out, mask=k_mask[:, None])

# 2c02d9cc: (T([1000,2048], bf16), T([256], bf16), T([1,1000,256], bf16), T([1,1000,256], bf16), T([1000,1024], bf16), T([256], bf16), S([1,1000,2048]), S([1,1000,-1,256]), S([1,1000,1024]), S([1,1000,-1,256]), S([1,4,2,1000,256]), S([1,8,1000,256]))
@oracle_impl(hardware="B200", point="2c02d9cc", BLOCK_ROWS=2, num_warps=4)
def oracle_forward(inputs, *, BLOCK_ROWS: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs[:6]

    query = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ_LEN, HEAD_DIM),
        (Q_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    key = torch.empty_strided(
        (BATCH, KV_HEADS, SEQ_LEN, HEAD_DIM),
        (KV_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, KV_HEADS * HEAD_DIM, 1),
        device=arg4_1.device,
        dtype=torch.bfloat16,
    )
    repeated_key = torch.empty(
        (BATCH, Q_HEADS, SEQ_LEN, HEAD_DIM),
        device=arg4_1.device,
        dtype=torch.bfloat16,
    )

    _gemma_rmsnorm_rope_fused_kernel[(triton.cdiv(KV_ROWS, BLOCK_ROWS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
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
    return query, key, repeated_key
