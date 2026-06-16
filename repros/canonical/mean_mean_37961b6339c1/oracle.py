"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Qwen bf16 query/key RMSNorm plus precomputed-RoPE scope, including the returned non-contiguous query RoPE layout, the returned non-contiguous key RoPE layout, and the grouped-KV expand/clone/view materialization into contiguous query-head layout. Inductor currently schedules the two fixed-width RMSNorm reductions and the RoPE/repeat materialization as generic reduction and pointwise/layout work; the missing optimization is SCHEDULER_FUSION, sinking the bf16 RMSNorm/RoPE epilogue and grouped-query repeat stores into fixed-shape producer kernels while preserving the repro's dtype rounding and output strides."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


HEAD_DIM = 128
HALF_DIM = 64
EPS = 1.0e-6


@triton.jit
def _round_bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


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
def _query_rmsnorm_rope_kernel(
    mm_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    eps_value: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_D)
    row_mask = rows < n_rows
    mask = row_mask[:, None]

    base = rows[:, None] * BLOCK_D + cols[None, :]
    x = tl.load(mm_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x_for_reduce = tl.where(mask, x, 0.0)
    sum_sq = tl.sum(_f32_mul(x_for_reduce, x_for_reduce), axis=1)
    inv_rms = libdevice.rsqrt(_f32_add(_f32_mul(sum_sq, 1.0 / BLOCK_D), eps_value))

    rot_cols = tl.where(cols < BLOCK_HALF, cols + BLOCK_HALF, cols - BLOCK_HALF)
    x_rot = tl.load(
        mm_ptr + rows[:, None] * BLOCK_D + rot_cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    rot_weight = tl.load(weight_ptr + rot_cols).to(tl.float32)
    value = _round_bf16(_f32_mul(_round_bf16(_f32_mul(x, inv_rms[:, None])), weight[None, :]))
    rotated = _round_bf16(_f32_mul(_round_bf16(_f32_mul(x_rot, inv_rms[:, None])), rot_weight[None, :]))
    rotated = rotated * tl.where(cols < BLOCK_HALF, -1.0, 1.0)[None, :]

    batch = rows // (seq_len * num_heads)
    pos = (rows // num_heads) - batch * seq_len
    rope_base = pos[:, None] * BLOCK_D + cols[None, :]
    cos_value = tl.load(cos_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)
    sin_value = tl.load(sin_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)

    out = _f32_add(_round_bf16(_f32_mul(value, cos_value)), _round_bf16(_f32_mul(rotated, sin_value)))
    tl.store(out_ptr + base, out, mask=mask)


@triton.jit
def _key_rmsnorm_rope_repeat_kernel(
    mm_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    key_out_ptr,
    repeat_out_ptr,
    n_rows: tl.constexpr,
    num_heads: tl.constexpr,
    out_heads: tl.constexpr,
    repeat: tl.constexpr,
    seq_len: tl.constexpr,
    eps_value: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_D)
    row_mask = rows < n_rows
    mask = row_mask[:, None]

    base = rows[:, None] * BLOCK_D + cols[None, :]
    x = tl.load(mm_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x_for_reduce = tl.where(mask, x, 0.0)
    sum_sq = tl.sum(_f32_mul(x_for_reduce, x_for_reduce), axis=1)
    inv_rms = libdevice.rsqrt(_f32_add(_f32_mul(sum_sq, 1.0 / BLOCK_D), eps_value))

    rot_cols = tl.where(cols < BLOCK_HALF, cols + BLOCK_HALF, cols - BLOCK_HALF)
    x_rot = tl.load(
        mm_ptr + rows[:, None] * BLOCK_D + rot_cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    rot_weight = tl.load(weight_ptr + rot_cols).to(tl.float32)
    value = _round_bf16(_f32_mul(_round_bf16(_f32_mul(x, inv_rms[:, None])), weight[None, :]))
    rotated = _round_bf16(_f32_mul(_round_bf16(_f32_mul(x_rot, inv_rms[:, None])), rot_weight[None, :]))
    rotated = rotated * tl.where(cols < BLOCK_HALF, -1.0, 1.0)[None, :]

    batch = rows // (seq_len * num_heads)
    pos = (rows // num_heads) - batch * seq_len
    rope_base = pos[:, None] * BLOCK_D + cols[None, :]
    cos_value = tl.load(cos_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)
    sin_value = tl.load(sin_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)

    out = _f32_add(_round_bf16(_f32_mul(value, cos_value)), _round_bf16(_f32_mul(rotated, sin_value)))
    tl.store(key_out_ptr + base, out, mask=mask)

    kv_head = rows - (rows // num_heads) * num_heads
    for repeat_idx in tl.static_range(0, repeat):
        out_head = kv_head * repeat + repeat_idx
        out_base = (
            batch[:, None] * (out_heads * seq_len * BLOCK_D)
            + out_head[:, None] * (seq_len * BLOCK_D)
            + pos[:, None] * BLOCK_D
            + cols[None, :]
        )
        tl.store(repeat_out_ptr + out_base, out, mask=mask)


# f8472361: Qwen3-0.6B query/key RMSNorm + RoPE, seq=1000, Hq=16, Hkv=8.
@oracle_impl(hardware="B200", point="f8472361", BLOCK_ROWS=4, num_warps=4)
def oracle_forward(inputs, *, BLOCK_ROWS: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4

    seq_len = int(arg2_1.shape[1])
    batch = int(arg0_1.shape[0]) // seq_len
    q_heads = int(arg0_1.shape[1]) // HEAD_DIM
    kv_heads = int(arg4_1.shape[1]) // HEAD_DIM
    repeat = q_heads // kv_heads

    query_rope = torch.empty_strided(
        (batch, q_heads, seq_len, HEAD_DIM),
        (q_heads * seq_len * HEAD_DIM, HEAD_DIM, q_heads * HEAD_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    key_rope = torch.empty_strided(
        (batch, kv_heads, seq_len, HEAD_DIM),
        (kv_heads * seq_len * HEAD_DIM, HEAD_DIM, kv_heads * HEAD_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    repeated_key = torch.empty(
        tuple(int(dim) for dim in _shape_param_5),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    query_rows = batch * seq_len * q_heads
    key_rows = batch * seq_len * kv_heads

    _query_rmsnorm_rope_kernel[(triton.cdiv(query_rows, BLOCK_ROWS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        query_rope,
        n_rows=query_rows,
        num_heads=q_heads,
        seq_len=seq_len,
        eps_value=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_D=HEAD_DIM,
        BLOCK_HALF=HALF_DIM,
        num_warps=num_warps,
        num_stages=3,
    )
    _key_rmsnorm_rope_repeat_kernel[(triton.cdiv(key_rows, BLOCK_ROWS),)](
        arg4_1,
        arg5_1,
        arg2_1,
        arg3_1,
        key_rope,
        repeated_key,
        n_rows=key_rows,
        num_heads=kv_heads,
        out_heads=q_heads,
        repeat=repeat,
        seq_len=seq_len,
        eps_value=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_D=HEAD_DIM,
        BLOCK_HALF=HALF_DIM,
        num_warps=num_warps,
        num_stages=3,
    )

    return query_rope, key_rope, repeated_key
